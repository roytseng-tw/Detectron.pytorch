from torch import nn

from core.config import cfg
from datasets import json_dataset
import roi_data.fast_rcnn


class GenerateProposalLabelsOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rpn_rois, roidb, im_info):
        """Op for generating training labels for RPN proposals. This is used
        when training RPN jointly with Fast/Mask R-CNN (as in end-to-end
        Faster R-CNN training).

        blobs_in:
          - 'rpn_rois': 2D tensor of RPN proposals output by GenerateProposals
          - 'roidb': roidb entries that will be labeled
          - 'im_info': See GenerateProposals doc.

        blobs_out:
          - (variable set of blobs): returns whatever blobs are required for
            training the model. It does this by querying the data loader for
            the list of blobs that are needed.
        """
        im_scales = im_info.data.numpy()[:, 2]

        output_blob_names = roi_data.fast_rcnn.get_fast_rcnn_blob_names()
        # For historical consistency with the original Faster R-CNN
        # implementation we are *not* filtering crowd proposals.
        # This choice should be investigated in the future (it likely does
        # not matter).
        # Note: crowd_thresh=0 will ignore _filter_crowd_proposals
        json_dataset.add_proposals(roidb, rpn_rois, im_scales, crowd_thresh=0)
        blobs = {k: [] for k in output_blob_names}
        roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)

        return blobs
