from model2.nodulenet.layer import *
from model2.nodulenet.config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import torch.nn.functional as F
from model2.nodulenet.utils.util import center_box_to_coord_box, ext2factor, clip_boxes, crop_boxes2mask_single
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm
from torch import nn
import torch


bn_momentum = 0.1
affine = True

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32))

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64))

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channeld dimension of coord
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(128, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


    def forward(self, x):
        out = self.preBlock(x)#16
        out_pool = out
        out1 = self.forw1(out_pool)#32
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        #out2 = self.drop(out2)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        #out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2), 1))#64+64

        return x, out1, comb2, out2
        # return [x, out1, comb2], out2

class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        

        return logits, deltas

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas

class MaskHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(MaskHead, self).__init__()
        self.num_class = cfg['num_class']

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back2 = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))
        
        self.back3 = nn.Sequential(
            nn.Conv3d(65, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64, momentum=bn_momentum, affine=affine),
            nn.ReLU(inplace = True))

        for i in range(self.num_class):
            setattr(self, 'logits' + str(i + 1), nn.Conv3d(64, 1, kernel_size=1))

    def forward(self, crop_f4, crop_f2, im):
        # crop_f2.unsqueeze(0)
        up2 = self.up2(crop_f4)
        up2 = self.back2(torch.cat((up2, crop_f2), 1))
        up3 = self.up3(up2)
        up3 = self.back3(torch.cat((up3, im), 1))

        logits = getattr(self, 'logits1')(up3)
        logits = logits.squeeze()
        out_mask = torch.sigmoid(logits)>0.5
        out_mask = out_mask.int()
        out_mask = out_mask.unsqueeze(0)
        out_mask = out_mask.unsqueeze(0)
        return out_mask


    def forwards(self, detections, features):
        img, f_2, f_4 = features  

        # Squeeze the first dimension to recover from protection on avoiding split by dataparallel      
        img = img.squeeze(0)
        f_2 = f_2.squeeze(0)
        f_4 = f_4.squeeze(0)

        _, _, D, H, W = img.shape
        out = []
        # N = detections.shape[0]
        # out = Variable(torch.zeros((N, D, H, W))).cuda()
        mask_probs_for_nms = []
        for idx, detection in enumerate(detections):
            b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection

            up1 = f_4[
                b,
                :, 
                torch.div(z_start, 4, rounding_mode='floor'):torch.div(z_end, 4, rounding_mode='floor'), 
                torch.div(y_start, 4, rounding_mode='floor'):torch.div(y_end, 4, rounding_mode='floor'), 
                torch.div(x_start, 4, rounding_mode='floor'):torch.div(x_end, 4, rounding_mode='floor'), 
            ].unsqueeze(0)
            up2 = self.up2(up1)
            up2 = self.back2(torch.cat((up2, f_2[
                b, 
                :, 
                torch.div(z_start, 2, rounding_mode='floor'):torch.div(z_end, 2, rounding_mode='floor'), 
                torch.div(y_start, 2, rounding_mode='floor'):torch.div(y_end, 2, rounding_mode='floor'), 
                torch.div(x_start, 2, rounding_mode='floor'):torch.div(x_end, 2, rounding_mode='floor'), 
            ].unsqueeze(0)), 1))
            up3 = self.up3(up2)
            im = img[b, :, z_start:z_end, y_start:y_end, x_start:x_end].unsqueeze(0)
            up3 = self.back3(torch.cat((up3, im), 1))

            logits = getattr(self, 'logits' + str(int(cat)))(up3)
            logits = logits.squeeze()
            out_mask = torch.sigmoid(logits)>0.5
            out_mask = out_mask.int()
            # out_mask = torch.where(torch.sigmoid(logits)>0.5)
 
            # # TODO: cause GPU run out of memory
            # # This is not workable in training -->
            # # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            # # mask = Variable(torch.zeros((D, H, W))).cuda()
            # mask = Variable(torch.zeros((D, H, W)))
            # mask[z_start:z_end, y_start:y_end, x_start:x_end] = out_mask
            # mask = mask.unsqueeze(0)
            # out.append(mask)
            # # out.append(torch.where(mask))
            
            # # TODO: cause GPU run out of memory
            # # This is not workable in training -->
            # # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            # # mask = Variable(torch.zeros((D, H, W))).cuda()
            # # if idx > 1: break
            
        # out = torch.cat(out, 0)
        # return out
            # TODO: bad implementation about shap slicing
            mask_probs_for_nms.append((out_mask, features[0][0, 0, 0].shape))
        return mask_probs_for_nms
        
# TODO: use crop_boxes
# def crop_mask_regions(masks, crop_boxes, out_shape):
#     out_mask = torch.zeros(out_shape)
#     for i in range(len(crop_boxes)):
#         post_indices = torch.stack(masks[i], dim=0)
#         b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
#         out_mask[(
#             torch.arange(out_mask.shape[0]), torch.arange(out_mask.shape[1]), 
#             post_indices[0], post_indices[1], post_indices[2]
#         )] = i+1
#     return out_mask


def crop_mask_regions(masks, crop_boxes):
    out = []
    for i in range(len(crop_boxes)):
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = crop_boxes[i]
        m = masks[i][z_start:z_end, y_start:y_end, x_start:x_end].contiguous()
        out.append(m)
    return out


def top1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res

def random1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res


class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size'] 

    def forward(self, f, inputs, proposals):
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = p[0].int()
            center = p[2:5]
            side_length = p[5:8]
            c0 = center - side_length / 2 # left bottom corner
            c1 = c0 + side_length # right upper corner
            c0 = (c0 / self.scale).floor().long()
            c1 = (c1 / self.scale).ceil().long()
            minimum = torch.LongTensor([0, 0, 0])
            maximum = torch.LongTensor([self.DEPTH, self.HEIGHT, self.WIDTH])
            maximum = maximum / self.scale
            # maximum = torch.LongTensor(
            #     np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale)
            # minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            # maximum = torch.LongTensor(
            #     np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

            c0 = torch.stack((c0, minimum), 0)
            c1 = torch.stack((c1, maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            # TODO:
            c0 = np.int32(c0.cpu().data.numpy())
            c1 = np.int32(c1.cpu().data.numpy())

            # TODO: should get correct c0, c1 at first to avoid if-else statement
            # # Slice 0 dim, should never happen
            # if torch.any((c1 - c0).cpu().data.numpy() < 1):
            #     print(p)
            #     print('c0:', c0, ', c1:', c1)
            #     # TODO: Leon
            #     continue
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class NoduleNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(NoduleNet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config, 1, 128)
        self.rpn = RpnHead(config, in_channels=128)
        self.rcnn_head = RcnnHead(config, in_channels=64)
        self.rcnn_crop = CropRoi(self.cfg, cfg['rcnn_crop_size'])
        self.mask_head = MaskHead(config, in_channels=128)
        self.use_rcnn = False
        self.use_mask = False

        # self.rpn_loss = Loss(cfg['num_hard'])
        

    def ori_forward(self, inputs, truth_boxes, truth_labels, truth_masks, masks, split_combiner=None, nzhw=None):
        features, feat_4 = data_parallel(self.feature_net, (inputs)); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

        b,D,H,W,_,num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1) #print('rpn_logit ', self.rpn_logits_flat.shape)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6) #print('rpn_delta ', self.rpn_deltas_flat.shape)


        self.rpn_window    = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn or self.mode in ['eval', 'test']:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)
            # print 'length of rpn proposals', self.rpn_proposals.shape

        if self.mode in ['train', 'valid']:
            # self.rpn_proposals = torch.zeros((0, 8)).cuda()
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(self.cfg, self.mode, inputs, self.rpn_window, truth_boxes, truth_labels )

            if self.use_rcnn:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                # print('aaaaaaaa',  self.rpn_proposals)
                self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                    make_rcnn_target(self.cfg, self.mode, inputs, self.rpn_proposals,
                        truth_boxes, truth_labels, truth_masks)
                # print('bbbbbbbb',  self.rpn_proposals)
        
        #rcnn proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

        self.mask_probs, self.mask_targets = [], []
        self.crop_boxes = []
        if self.use_rcnn:
            if self.rpn_proposals.shape[0] > 0:
                rcnn_crops = self.rcnn_crop(feat_4, inputs, self.rpn_proposals)
                self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, inputs, self.rpn_proposals, 
                                                                        self.rcnn_logits, self.rcnn_deltas)

            if self.mode in ['eval']:
                # Ensemble
                fpr_res = get_probability(self.cfg, self.mode, inputs, self.rpn_proposals,  self.rcnn_logits, self.rcnn_deltas)
                self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2

            if self.use_mask and len(self.detections):
                # keep batch index, z, y, x, d, h, w, class
                if len(self.detections):
                    self.crop_boxes = self.detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                    self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
                    self.crop_boxes = self.crop_boxes.astype(np.int32)
                    # self.crop_boxes[:, 4:-1] = self.crop_boxes[:, 4:-1] + np.ones_like(self.crop_boxes[:, 4:-1])
                    self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 4)
                    self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
                
                # if self.mode in ['eval', 'test']:
                #     self.crop_boxes = top1pred(self.crop_boxes)
                # else:
                #     self.crop_boxes = random1pred(self.crop_boxes)

                if self.mode in ['train', 'valid']:
                    self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
                        truth_boxes, truth_labels, masks)
                    # print('xxx', self.mask_targets)

                # Make sure to keep feature maps not splitted by data parallel
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))

                if self.mode in ['eval', 'test']:
                    mask_keep, self.mask_probs = mask_nms(self.cfg, self.mode, self.mask_probs, self.crop_boxes, inputs)
                #    self.crop_boxes = torch.index_select(self.crop_boxes, 0, mask_keep)
                #    self.detections = torch.index_select(self.detections, 0, mask_keep)
                #    self.mask_probs = torch.index_select(self.mask_probs, 0, mask_keep)
                    self.crop_boxes = self.crop_boxes[mask_keep]
                    self.detections = self.detections[mask_keep]
                    # self.mask_probs = self.mask_probs[mask_keep]
                
                self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)

    def inference(self, inputs):
        # Image feature
        features, feat_4 = self.feature_net(inputs); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        # RPN proposals
        self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)

        b,D,H,W,_,num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1);#print('rpn_logit ', self.rpn_logits_flat.shape)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6);#print('rpn_delta ', self.rpn_deltas_flat.shape)

        self.rpn_window  = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)
            # print 'length of rpn proposals', self.rpn_proposals.shape

        # RCNN proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.mask_probs, self.mask_targets = [], []
        self.crop_boxes = []
        if self.use_rcnn:
            if len(self.rpn_proposals) > 0:
                rcnn_crops = self.rcnn_crop(feat_4, inputs, self.rpn_proposals)
                self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
                self.detections, self.keeps = rcnn_nms(
                    self.cfg, self.mode, inputs, self.rpn_proposals, 
                    self.rcnn_logits, self.rcnn_deltas
                ) 

            # pred_mask = np.zeros(list(inputs.shape[2:]))
            if self.use_mask and len(self.detections):
                # keep batch index, z, y, x, d, h, w, class
                if len(self.detections):
                    self.crop_boxes = self.detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                    self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
                    self.crop_boxes = self.crop_boxes.astype(np.int32)
                    # self.crop_boxes[:, 4:-1] = self.crop_boxes[:, 4:-1] + np.ones_like(self.crop_boxes[:, 4:-1])
                    self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 4)
                    self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
                
                # Make sure to keep feature maps not splitted by data parallel
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                self.mask_probs = self.mask_head(torch.from_numpy(self.crop_boxes).cuda(), features)

                mask_keep = mask_nms(self.cfg, self.mode, self.mask_probs, self.crop_boxes, inputs)
                self.crop_boxes = self.crop_boxes[mask_keep]
                self.detections = self.detections[mask_keep]
                # self.mask_probs = self.mask_probs[mask_keep]
                out_masks = []
                for keep_idx in mask_keep:
                    out_masks.append(self.mask_probs[keep_idx])
                self.mask_probs = out_masks
                
                pred_mask = crop_mask_regions(self.mask_probs, self.crop_boxes, features[0][0].shape)

                # # segments = [torch.sigmoid(m).cpu().numpy() > 0.5 for m in self.mask_probs]
                # segments = [torch.sigmoid(m) > 0.5 for m in self.mask_probs]
                # pred_mask = crop_boxes2mask_single(self.crop_boxes[:, 1:], segments, inputs.shape[2:])
        # TODO: correctly get the num_class and batch size dimension
        # pred_mask = pred_mask[None, None]
        return pred_mask

    def forward(self, inputs):
        """This version try to eliminate TraceWarning in ONNX"""
        # Image feature
        features, feat_4 = self.feature_net(inputs); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        # RPN proposals
        self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn(fs)

        b,D,H,W,_,num_class = self.rpn_logits_flat.shape

        self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1);#print('rpn_logit ', self.rpn_logits_flat.shape)
        self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6);#print('rpn_delta ', self.rpn_deltas_flat.shape)

        self.rpn_window    = make_rpn_windows(fs, self.cfg)
        self.rpn_proposals = []
        if self.use_rcnn:
            self.rpn_proposals = rpn_nms(self.cfg, self.mode, inputs, self.rpn_window,
                  self.rpn_logits_flat, self.rpn_deltas_flat)
            # print 'length of rpn proposals', self.rpn_proposals.shape

        # RCNN proposals
        self.detections = copy.deepcopy(self.rpn_proposals)
        self.mask_probs, self.mask_targets = [], []
        self.crop_boxes = []
        if self.use_rcnn:
            # TODO: pass even no rpn_proposals
            # if len(self.rpn_proposals) > 0:
            rcnn_crops = self.rcnn_crop(feat_4, inputs, self.rpn_proposals)
            self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
            self.detections, self.keeps = rcnn_nms(
                self.cfg, self.mode, inputs, self.rpn_proposals, 
                self.rcnn_logits, self.rcnn_deltas
            )

            # pred_mask = np.zeros(list(inputs.shape[2:]))
            # if self.use_mask and len(self.detections):
            # TODO: pass even no proposal in self.detections
            if self.use_mask:
                # keep batch index, z, y, x, d, h, w, class
                if len(self.detections):
                    self.crop_boxes = self.detections[:, [0, 2, 3, 4, 5, 6, 7, 8]].cpu().numpy().copy()
                    self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
                    self.crop_boxes = self.crop_boxes.astype(np.int32)
                    # self.crop_boxes[:, 4:-1] = self.crop_boxes[:, 4:-1] + np.ones_like(self.crop_boxes[:, 4:-1])
                    self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 4)
                    self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
                
                # Make sure to keep feature maps not splitted by data parallel
                features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
                self.mask_probs = self.mask_head(torch.from_numpy(self.crop_boxes), features)
                # self.mask_probs = self.mask_head(torch.from_numpy(self.crop_boxes).cuda(), features)

                mask_keep, self.mask_probs = mask_nms(self.cfg, self.mode, self.mask_probs, self.crop_boxes, inputs)
                self.crop_boxes = self.crop_boxes[mask_keep]
                self.detections = self.detections[mask_keep]
                # self.mask_probs = self.mask_probs[mask_keep]
                # out_masks = []
                # for keep_idx in mask_keep:
                #     out_masks.append(self.mask_probs[keep_idx])
                # self.mask_probs = out_masks
                
                mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)
                segments = [torch.sigmoid(m) > 0.5 for m in mask_probs]
                pred_mask = crop_boxes2mask_single(self.crop_boxes[:, 1:], segments, inputs.shape[2:])
    
                # pred_mask = crop_mask_regions(self.mask_probs, self.crop_boxes, features[0][0].shape)

                # # segments = [torch.sigmoid(m).cpu().numpy() > 0.5 for m in self.mask_probs]
                # segments = [torch.sigmoid(m) > 0.5 for m in self.mask_probs]
                # pred_mask = crop_boxes2mask_single(self.crop_boxes[:, 1:], segments, inputs.shape[2:])
        # TODO: correctly get the num_class and batch size dimension
        # pred_mask = pred_mask[None, None]
        return pred_mask

    def forward2(self, inputs, bboxes):
        features = data_parallel(self.feature_net, (inputs)) #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        self.crop_boxes = []
        for b in range(len(bboxes)):
            self.crop_boxes.append(np.column_stack((np.zeros((len(bboxes[b]) + b, 1)), bboxes[b])))

        self.crop_boxes = np.concatenate(self.crop_boxes, 0)
        self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
        self.crop_boxes = self.crop_boxes.astype(np.int32)
        self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 8)
        self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
#         self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
#             truth_boxes, truth_labels, masks)

        # Make sure to keep feature maps not splitted by data parallel
        features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
        self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))
        self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)

    def forward_mask(self, inputs, truth_boxes, truth_labels, truth_masks, masks, split_combiner=None, nzhw=None):
        features, feat_4 = data_parallel(self.feature_net, (inputs)); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]

        # keep batch index, z, y, x, d, h, w, class
        self.crop_boxes = []
        for b in range(len(truth_boxes)):
            self.crop_boxes.append(np.column_stack((np.zeros((len(truth_boxes[b]) + b, 1)), truth_boxes[b], truth_labels[b])))
        self.crop_boxes = np.concatenate(self.crop_boxes, 0)
        self.crop_boxes[:, 1:-1] = center_box_to_coord_box(self.crop_boxes[:, 1:-1])
        self.crop_boxes = self.crop_boxes.astype(np.int32)
        self.crop_boxes[:, 1:-1] = ext2factor(self.crop_boxes[:, 1:-1], 4)
        self.crop_boxes[:, 1:-1] = clip_boxes(self.crop_boxes[:, 1:-1], inputs.shape[2:])
    
        # if self.mode in ['eval', 'test']:
        #     self.crop_boxes = top1pred(self.crop_boxes)
        # else:
        #     self.crop_boxes = random1pred(self.crop_boxes)

        if self.mode in ['train', 'valid']:
            self.mask_targets = make_mask_target(self.cfg, self.mode, inputs, self.crop_boxes,
                truth_boxes, truth_labels, masks)

        # Make sure to keep feature maps not splitted by data parallel
        features = [t.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1, -1, -1, -1) for t in features]
        self.mask_probs = data_parallel(self.mask_head, (torch.from_numpy(self.crop_boxes).cuda(), features))
        
        # if self.mode in ['eval', 'test']:
        #     mask_keep = mask_nms(self.cfg, self.mode, self.mask_probs, self.crop_boxes, inputs)
        # #    self.crop_boxes = torch.index_select(self.crop_boxes, 0, mask_keep)
        # #    self.detections = torch.index_select(self.detections, 0, mask_keep)
        # #    self.mask_probs = torch.index_select(self.mask_probs, 0, mask_keep)
        #     self.crop_boxes = self.crop_boxes[mask_keep]
        #     self.detections = self.detections[mask_keep]
        #     self.mask_probs = self.mask_probs[mask_keep]
        
        self.mask_probs = crop_mask_regions(self.mask_probs, self.crop_boxes)

    def loss(self, targets=None):
        cfg  = self.cfg
    
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rcnn_stats = None
        mask_stats = None

        self.mask_loss = torch.zeros(1).cuda()
    
        self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
           rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
            self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)

        if self.use_mask:
            self.mask_loss, mask_losses = mask_loss(self.mask_probs, self.mask_targets)
            mask_stats = [[] for _ in range(cfg['num_class'] - 1)] 
            for i in range(len(self.crop_boxes)):
                cat = int(self.crop_boxes[i][-1]) - 1
                mask_stats[cat].append(mask_losses[i])
            mask_stats = [np.mean(e) for e in mask_stats]
            mask_stats = np.array(mask_stats)
            mask_stats[mask_stats == 0] = np.nan
    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss \
                          + self.mask_loss

    
        return self.total_loss, rpn_stats, rcnn_stats, mask_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

    def set_anchor_params(self, anchor_ids, anchor_params):
        self.anchor_ids = anchor_ids
        self.anchor_params = anchor_params

    def crf(self, detections):
        """
        detections: numpy array of detection results [b, z, y, x, d, h, w, p]
        """
        res = []
        config = self.cfg
        anchor_ids = self.anchor_ids
        anchor_params = self.anchor_params
        anchor_centers = []

        for a in anchor_ids:
            # category starts from 1 with 0 denoting background
            # id starts from 0
            cat = a + 1
            dets = detections[detections[:, -1] == cat]
            if len(dets):
                b, p, z, y, x, d, h, w, _ = dets[0]
                anchor_centers.append([z, y, x])
                res.append(dets[0])
            else:
                # Does not have anchor box
                return detections
        
        pred_cats = np.unique(detections[:, -1]).astype(np.uint8)
        for cat in pred_cats:
            if cat - 1 not in anchor_ids:
                cat = int(cat)
                preds = detections[detections[:, -1] == cat]
                score = np.zeros((len(preds),))
                roi_name = config['roi_names'][cat - 1]

                for k, params in enumerate(anchor_params):
                    param = params[roi_name]
                    for i, det in enumerate(preds):
                        b, p, z, y, x, d, h, w, _ = det
                        d = np.array([z, y, x]) - np.array(anchor_centers[k])
                        prob = norm.pdf(d, param[0], param[1])
                        prob = np.log(prob)
                        prob = np.sum(prob)
                        score[i] += prob

                res.append(preds[score == score.max()][0])
            
        res = np.array(res)
        return res

if __name__ == '__main__':
    net = FasterRcnn(config)

    input = torch.rand([4,1,128,128,128])
    input = Variable(input)
