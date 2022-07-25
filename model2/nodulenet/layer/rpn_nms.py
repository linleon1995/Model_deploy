import numpy as np
from model2.nodulenet.layer.util import box_transform, box_transform_inv, clip_boxes
import itertools
import torch.nn.functional as F
import torch
from torch.autograd import Variable
try:
    from model2.nodulenet.utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from model2.nodulenet.utils.util import py_nms as torch_nms
    from model2.nodulenet.utils.util import py_box_overlap as torch_overlap


def make_rpn_windows_np(f, cfg):
    """
    Generating anchor boxes at each voxel on the feature map,
    the center of the anchor box on each voxel corresponds to center
    on the original input image.

    return
    windows: list of anchor boxes, [z, y, x, d, h, w]
    """
    stride = cfg['stride']
    anchors = np.asarray(cfg['anchors'])
    offset = (float(stride) - 1) / 2
    _, _, D, H, W = f.shape
    oz = np.arange(offset, offset + stride * (D - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (H - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (W - 1) + 1, stride)

    windows = []
    for z, y , x , a in itertools.product(oz, oh , ow , anchors):
        windows.append([z, y, x, a[0], a[1], a[2]])
    windows = np.array(windows)

    return windows


def make_rpn_windows(f, cfg):
    """
    Generating anchor boxes at each voxel on the feature map,
    the center of the anchor box on each voxel corresponds to center
    on the original input image.

    return
    windows: list of anchor boxes, [z, y, x, d, h, w]
    """
    stride = cfg['stride']
    # anchors = np.asarray(cfg['anchors'])
    anchors = torch.FloatTensor(cfg['anchors'])
    offset = (float(stride) - 1) / 2
    _, _, D, H, W = f.shape
    oz = torch.arange(offset, offset + stride * (D - 1) + 1, stride)
    oh = torch.arange(offset, offset + stride * (H - 1) + 1, stride)
    ow = torch.arange(offset, offset + stride * (W - 1) + 1, stride)

    x1 = torch.meshgrid([oz, oh, ow, anchors[:,0]])
    x2 = torch.meshgrid([oz, oh, ow, anchors[:,1]])
    x3 = torch.meshgrid([oz, oh, ow, anchors[:,2]])

    x1 = [x.contiguous().view(-1) for x in x1]
    anchors1 = x2[-1].contiguous().view(-1)
    anchors2 = x3[-1].contiguous().view(-1)

    x1.extend([anchors1, anchors2])
    windows = torch.stack(x1, dim=1)
    return windows


def rpn_nms(cfg, mode, inputs, window, logits_flat, deltas_flat):
    if mode in ['train',]:
        nms_pre_score_threshold = cfg['rpn_train_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_train_nms_overlap_threshold']

    elif mode in ['eval', 'valid', 'test',]:
        nms_pre_score_threshold = cfg['rpn_test_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_test_nms_overlap_threshold']

    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)


    logits = torch.sigmoid(logits_flat)
    deltas = deltas_flat
    # logits = torch.sigmoid(logits_flat).data.cpu().numpy()
    # deltas = deltas_flat.data.cpu().numpy()
    batch_size, _, depth, height, width = inputs.size()

    proposals = []
    for b in range(batch_size):
        proposal = [torch.empty((0, 8)).float()]
        # proposal = [torch.empty((0, 8),np.float32),]

        # ps = logits[b, : , 0].reshape(-1, 1)
        # ds = deltas[b, :, :]
        ps = logits[b, : , 0].reshape(-1, 1).cuda()
        ds = deltas[b, :, :].cuda()
        window = window.cuda()

        # Only those anchor boxes larger than a pre-defined threshold
        # will be chosen for nms computation
        index = torch.where(ps[:, 0] > nms_pre_score_threshold)[0]
        # TODO: pass enven no proposal exist
        # if len(index) > 0:
        p = ps[index]
        d = ds[index]
        w = window[index]
        box = rpn_decode(w, d, cfg['box_reg_weight'])
        box = clip_boxes(box, inputs.shape[2:])

        box = box.cuda()
        output = torch.cat((p, box), 1)
        # output = np.concatenate((p, box),1)

        # output = torch.from_numpy(output)
        
        # TODO: torch_nms casue two ONNX TracerWarning, command this temporally.
        # 1. while order.shape[0] > 0: 2. return dets[keep], torch.LongTensor(keep)
        # output, keep = torch_nms(output, nms_overlap_threshold)

        prop = torch.zeros((output.shape[0], 8)).float()
        # prop = np.zeros((output.shape[0], 8),np.float32)
        prop[:, 0] = b
        prop[:, 1:8] = output
        
        proposal.append(prop)

        proposal = torch.vstack(proposal)
        # proposal = np.vstack(proposal)
        proposals.append(proposal)

    proposals = torch.vstack(proposals)
    # proposals = np.vstack(proposals)

    # TODO: pass enven no proposal exist
    # # Just in case if there is no proposal, we still return a Tensor,
    # # torch.from_numpy() cannot take input with 0 dim
    # if proposals.shape[0] != 0:
    #     proposals = Variable(proposals)
    #     # proposals = Variable(torch.from_numpy(proposals)).cuda()
    #     return proposals
    # else:
    #     return Variable(torch.rand([0, 8]))
    #     # return Variable(torch.rand([0, 8])).cuda()

    return proposals

def rpn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)

def rpn_decode(window, delta, weight):
    return box_transform_inv(window, delta, weight)
