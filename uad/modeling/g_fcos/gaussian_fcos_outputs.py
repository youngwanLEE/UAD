import logging
import torch
import torch.nn.functional as F

import numpy as np

from detectron2.layers import cat
from detectron2.structures import Instances, Boxes, pairwise_iou
from detectron2.utils.comm import get_world_size
from fvcore.nn import sigmoid_focal_loss_jit

from uad.utils.comm import reduce_sum
from uad.layers import ml_nms

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:
    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization
Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 
    ctrness_pred: predicted centerness scores
"""

def gaussian_dist_pdf(val, mean, var):
    return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


def fcos_losses(
        labels,
        reg_targets,
        logits_pred,
        reg_pred,
        sigma_pred,
        sigma_weight,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,
        gt_inds,
):
    num_classes = logits_pred.size(1)
    labels = labels.flatten()

    pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
    num_pos_local = pos_inds.numel()
    num_gpus = get_world_size()
    total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
    num_pos_avg = max(total_num_pos / num_gpus, 1.0)


    reg_pred = reg_pred[pos_inds]
    reg_targets = reg_targets[pos_inds]
    pos_sigma_pred = sigma_pred[pos_inds]
    gt_inds = gt_inds[pos_inds]


    if pos_inds.numel() > 0:

        reg_loss, iou_target = iou_loss(
            reg_pred,
            reg_targets,
            iou_weight=True,
            iou_target=True
        # ) / loss_denorm
        )
        loss_denorm = max(reduce_sum(iou_target.sum()).item() / num_gpus, 1e-6)
        reg_loss /= loss_denorm

        ### uncertainty loss ###
        sigma_loss = - iou_target[:, None] * torch.log(
            gaussian_dist_pdf(reg_targets, reg_pred, pos_sigma_pred) + 1e-9
        )
        sigma_loss = sigma_loss.sum() / num_pos_avg
        sigma_loss *= sigma_weight


    else:
        reg_loss = reg_pred.sum() * 0
        # ctrness_loss = ctrness_pred.sum() * 0
        sigma_loss = sigma_pred.sum() * 0


    ### classification loss ###
    # TODO : check whether .clone().detach()
    # TODO : make a uncertainty-aware focal loss .py script
    # TODO : scaling sigma_pred
    class_target = torch.zeros_like(logits_pred)
    # class_target[pos_inds, labels[pos_inds]] = 1
    class_target[pos_inds, labels[pos_inds]] = iou_target.clone().detach()
    logits_pred_sigmoid = logits_pred.sigmoid()
    sigma = sigma_pred.clone().detach().mean(dim=-1).unsqueeze(-1)

    class_weight = (1 - sigma) * \
                   (class_target > 0.0).float() + \
        focal_loss_alpha * logits_pred_sigmoid.pow(focal_loss_gamma) * (class_target == 0.0).float()


    class_loss = F.binary_cross_entropy_with_logits(
        logits_pred,
        class_target,
        reduction="none",
    ) * class_weight

    class_loss = class_loss.sum() / num_pos_avg



    losses = {
        "loss_fcos_cls": class_loss,
        "loss_fcos_loc": reg_loss,
        "loss_fcos_sigma": sigma_loss

    }
    extras = {
        "pos_inds": pos_inds,
        "gt_inds": gt_inds,
        "loss_denorm": loss_denorm
    }
    return losses, extras


class GaussianFCOSOutputs(object):
    def __init__(
            self,
            images,
            locations,
            logits_pred,
            reg_pred,
            sigma_pred,
            sigma_weight,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            thresh_with_ctr,
            gt_instances=None,
            atss_topk=None,
            atss_anchor_sizes=None
    ):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        self.sigma_pred = sigma_pred
        self.locations = locations
        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr
        self.sigma_weight = sigma_weight
        # ywlee
        self.atss_topk = atss_topk
        self.atss_anchor_sizes = atss_anchor_sizes

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        if self.atss_topk is not None:
            training_targets = self.compute_targets_for_locations_by_atss(
                locations, self.gt_instances, self.atss_topk, self.atss_anchor_sizes)
        else:
            training_targets = self.compute_targets_for_locations(
                locations, self.gt_instances, loc_to_size_range
            )

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets": reg_targets,
            "target_inds": target_inds}


    def compute_targets_for_locations_by_atss(self, locations, targets, topk, anchor_sizes):
        labels = []
        reg_targets = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            # Selecting candidates based on the center distance between anchor point and gt_box
            gt_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2 # gt_cx : (#instance, )
            gt_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2 # gt_cy : (#instance, )
            gt_points = torch.stack((gt_cx, gt_cy), dim=1) # gt_points : (#instances, 2)
            anchor_points = torch.stack((xs, ys), dim=1) # anchor_points : (#loc, 2)
            distances = (anchor_points[:,None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt() # distances : (#loc, #instance)

            candidate_idxs = []
            beg = 0
            for level, num_loc in enumerate(self.num_loc_list):
                end = beg + num_loc
                distance_per_level = distances[beg:end, :] # distance_per_level : (#loc_per_level, #instance)
                topk = min(self.atss_topk, num_loc)
                _, topk_idxs_per_level = distance_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + beg)
                beg = end
            candidate_idxs = torch.cat(candidate_idxs, dim=0) # candidate_idxs : (#cloc(=topk*levels), #inst)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            # Note : this IoU can be considered as 'hyperparameter',
            #        since 'appropriate' anchor sizes per feature level should be pre-defined!

            if anchor_sizes:
                anchor_boxes = []
                beg = 0
                for level, num_loc in enumerate(self.num_loc_list):
                    end = beg + num_loc
                    anchor_l = anchor_sizes[level] / 2
                    xmins = anchor_points[beg:end, 0] - anchor_l
                    ymins = anchor_points[beg:end, 1] - anchor_l
                    xmaxs = anchor_points[beg:end, 0] + anchor_l
                    ymaxs = anchor_points[beg:end, 1] + anchor_l
                    anchor_boxes_per_level = torch.stack([xmins, ymins, xmaxs, ymaxs], dim=1)
                    anchor_boxes.append(anchor_boxes_per_level)
                    beg = end
                anchor_boxes = torch.cat(anchor_boxes, dim=0) # anchor_boxes : (#loc, 4) // bboxes : (#inst, 4)
                ious = pairwise_iou(Boxes(anchor_boxes), Boxes(bboxes)) # ious : (#loc, #inst)

                candidate_ious = ious[candidate_idxs]
                candidate_ious = torch.stack([torch.diag(e, 0) for e in candidate_ious], dim=0) # candidate_ious : (#cloc, #inst)

                candidate_means_per_gt = candidate_ious.mean(dim=0) # candidate_means_per_gt : (#inst)
                candidate_devis_per_gt = candidate_ious.std(dim=0)  # candidate_devis_per_gt : (#inst)
                candidate_thres_per_gt = candidate_means_per_gt + candidate_devis_per_gt
                is_pos = candidate_ious > candidate_thres_per_gt # is_pos : (#cloc, #inst)

                temp = torch.zeros_like(distances) # (#loc, #inst)
                # debug
                temp[candidate_idxs * is_pos] = 1 # jerry-built...
                temp[0, :] = 0
                is_topk_boxes = temp > 0 # (#loc, #inst)

                # Limiting the final positive samples' center to object
                l = xs[:, None] - bboxes[:, 0][None]
                t = ys[:, None] - bboxes[:, 1][None]
                r = bboxes[:, 2][None] - xs[:, None]
                b = bboxes[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2) # (#loc, #inst, 4)

                # ATSS substitutes Center-Sampling of FCOS
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0 # (#loc, #inst)

            else: # old-implementation
                temp = torch.zeros_like(distances)
                temp[candidate_idxs] = 1
                is_topk_boxes = temp > 0

                # Limiting the final positive samples' center to object
                l = xs[:, None] - bboxes[:, 0][None]
                t = ys[:, None] - bboxes[:, 1][None]
                r = bboxes[:, 2][None] - xs[:, None]
                b = bboxes[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

                if self.center_sample:
                    is_in_boxes = self.get_sample_region(
                        bboxes, self.strides, self.num_loc_list,
                        xs, ys, radius=self.radius
                    )
                else:
                    is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0


            area = targets_per_im.gt_boxes.area() # (#inst,)
            locations_to_gt_area = area[None].repeat(len(locations), 1) # (#loc, #inst)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_topk_boxes == 0] = INF

            # ATSS
            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = ious.clone().detach()
            ious_inf[is_in_boxes == 0] = -INF
            ious_inf[is_topk_boxes == 0] = -INF
            locations_to_max_iou, locations_to_gt_inds = ious_inf.max(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_max_iou == -INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            target_inds.append(target_inds_per_im)



        return {"labels": labels,
                "reg_targets": reg_targets,
                "target_inds": target_inds}


    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.
        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets, gt_inds = (
            training_targets["labels"],
            training_targets["reg_targets"],
            training_targets["target_inds"])

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.logits_pred
            ], dim=0, )
        reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.reg_pred
            ], dim=0, )
        sigma_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4)
                for x in self.sigma_pred
            ], dim=0, )
        labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels
            ], dim=0, )

        gt_inds = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in gt_inds
            ], dim=0, )

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0, )

        return fcos_losses(
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            sigma_pred,
            self.sigma_weight,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            gt_inds
        )

    def predict_proposals(self, top_feats):
        sampled_boxes = []

        bundle = {
            "l": self.locations, "o": self.logits_pred,
            "r": self.reg_pred,
            "s": self.strides,  "v": self.sigma_pred
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, instance in enumerate(zip(*bundle.values())):
            instance_dict = dict(zip(bundle.keys(), instance))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = instance_dict["l"]
            o = instance_dict["o"]
            r = instance_dict["r"] * instance_dict["s"]
            v = instance_dict["v"]
            t = instance_dict["t"] if "t" in bundle else None

            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, v, self.image_sizes, t
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            # reg_pred, ctrness, sigma_pred,
            reg_pred,  sigma_pred,
            image_sizes, top_feat=None):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        sigma   = sigma_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        sigma   = sigma.reshape(N, -1, 4)

        #for vis
        # sigma_l = sigma[:,0]
        # sigma_t = sigma[:,1]
        # sigma_r = sigma[:,2]
        # sigma_b = sigma[:,3]
        #for vis
        uncertainty   = sigma.mean(dim=-1)

        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.thresh_with_ctr:
            box_cls = box_cls

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        if not self.thresh_with_ctr:
            box_cls = box_cls

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]

            ## uncertainty
            certainty = 1 - uncertainty
            per_box_certainty = certainty[i]
            per_box_certainty = per_box_certainty[per_box_loc]
            ##

            ## for 4-ways uncertainties visualization
            certainties = 1 - sigma
            per_box_certainties = certainties[i]
            per_box_certainties = per_box_certainties[per_box_loc]
            ## for vis

            per_locations = locations[per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxlist.pred_boxes = Boxes(detections)
            boxlist.scores = torch.sqrt(per_box_cls)
            # uncertainty
            boxlist.certainty = per_box_certainty
            boxlist.certainties = per_box_certainties
            # uncertainty
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results