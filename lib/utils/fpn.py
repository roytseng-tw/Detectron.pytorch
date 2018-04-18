import numpy as np

import utils.boxes as box_utils
from core.config import cfg


# ---------------------------------------------------------------------------- #
# Helper functions for working with multilevel FPN RoIs
# ---------------------------------------------------------------------------- #

def map_rois_to_fpn_levels(rois, k_min, k_max):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """
    # Compute level ids
    areas, neg_idx = box_utils.boxes_area(rois)
    areas[neg_idx] = 0  # np.sqrt will remove the entries with negative value
    s = np.sqrt(areas)
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    # Eqn.(1) in FPN paper
    target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
    target_lvls = np.clip(target_lvls, k_min, k_max)

    # Mark to discard negative area roi. See utils.fpn.add_multilevel_roi_blobs
    # target_lvls[neg_idx] = -1
    return target_lvls


def add_multilevel_roi_blobs(
        blobs, blob_prefix, rois, target_lvls, lvl_min, lvl_max
    ):
    """Add RoI blobs for multiple FPN levels to the blobs dict.

    blobs: a dict mapping from blob name to numpy ndarray
    blob_prefix: name prefix to use for the FPN blobs
    rois: the source rois as a 2D numpy array of shape (N, 5) where each row is
      an roi and the columns encode (batch_idx, x1, y1, x2, y2)
    target_lvls: numpy array of shape (N, ) indicating which FPN level each roi
      in rois should be assigned to. -1 means correspoind roi should be discarded.
    lvl_min: the finest (highest resolution) FPN level (e.g., 2)
    lvl_max: the coarest (lowest resolution) FPN level (e.g., 6)
    """
    rois_idx_order = np.empty((0, ))
    rois_stacked = np.zeros((0, 5), dtype=np.float32)  # for assert
    # target_lvls = remove_negative_area_roi_blobs(blobs, blob_prefix, rois, target_lvls)
    for lvl in range(lvl_min, lvl_max + 1):
        idx_lvl = np.where(target_lvls == lvl)[0]
        blobs[blob_prefix + '_fpn' + str(lvl)] = rois[idx_lvl, :]
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_stacked = np.vstack(
            [rois_stacked, blobs[blob_prefix + '_fpn' + str(lvl)]]
        )
    rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
    blobs[blob_prefix + '_idx_restore_int32'] = rois_idx_restore
    # Sanity check that restore order is correct
    assert (rois_stacked[rois_idx_restore] == rois).all()


def remove_negative_area_roi_blobs(blobs, blob_prefix, rois, target_lvls):
    """ Delete roi entries that have negative area (Uncompleted) """
    idx_neg = np.where(target_lvls == -1)[0]
    rois = np.delete(rois, idx_neg, axis=0)
    blobs[blob_prefix] = rois
    target_lvls = np.delete(target_lvls, idx_neg, axis=0)
    #TODO: other blobs in faster_rcnn.get_fast_rcnn_blob_names should also be modified
    return target_lvls
