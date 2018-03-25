"""
Helper functions for converting resnet pretrained weights from other formats
"""


def convert_state_dict(src_dict):
  """Return the correct mapping of tensor name and value

  Mapping from the names of torchvision model to our resnet conv_body and box_head.
  """
  dst_dict = {}
  for k, v in src_dict.items():
    toks = k.split('.')
    if k.startswith('layer'):
      assert len(toks[0]) == 6
      res_id = int(toks[0][5]) + 1
      name = '.'.join(['res%d' % res_id] + toks[1:])
      dst_dict[name] = v
    else:
      name = '.'.join(['res1'] + toks)
      dst_dict[name] = v
  return dst_dict
