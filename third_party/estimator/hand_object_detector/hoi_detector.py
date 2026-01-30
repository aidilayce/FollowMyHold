'''
Pipeline version of the demo.
'''
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from foho.configs import third_party_root

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_TP = third_party_root()
sys.path.append(os.path.join(_TP, "estimator"))
sys.path.append(os.path.join(_TP, "estimator", "hand_object_detector"))

import _init_paths
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
# from common_utils.hand_object_segmentation import obj_hand_bboxes, obj_mask_and_bbox, hand_segmentation #LISA is here

# from utils.ho_det_utils import union_box
def union_box(*bboxes):
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = (bboxes[:, 0] + bboxes[:, 2]).max()
    y2 = (bboxes[:, 1] + bboxes[:, 3]).max()    
    return np.array([x1, y1, x2-x1, y2-y1])


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save results',
                      default="images_det")
  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=89999, type=int, required=True)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0:
            img_obj_id.append(-1)
            continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
    return img_obj_id

class Config:
    def __init__(self):
        self.size = None
        self.color = None
        self.name = None


def hand_object_detector(image):
    from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
    from model.rpn.bbox_transform import clip_boxes
    # from model.nms.nms_wrapper import nms
    from model.roi_layers import nms
    from model.rpn.bbox_transform import bbox_transform_inv
    from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
    from model.utils.blob import im_list_to_blob
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet
    import pdb

    try:
        xrange          # Python 2
    except NameError:
        xrange = range  # Python 3

    args = Config()
    hod_root = os.path.join(_TP, "estimator", "hand_object_detector")
    args.cfg_file = os.path.join(hod_root, "cfgs", "res101.yml")
    args.dataset = 'pascal_voc'
    args.net = 'res101'
    args.set_cfgs = None
    args.load_dir = os.path.join(hod_root, "models")
    args.image_dir = os.path.join(hod_root, "images")
    args.save_dir = os.path.join(hod_root, "images_det")
    args.cuda = True
    args.class_agnostic = False
    args.webcam_num = -1
    args.parallel_type = 0
    args.checksession = 1
    args.checkepoch = 8
    args.checkpoint = 89999
    args.batch_size = 1
    args.vis = False
    args.thresh_hand = 0.1 # 0.5
    args.thresh_obj = 0.1 # 0.5

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception('There is no input directory for loading network from ' + model_dir)
    load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
    args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

    # initilize the network here.
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    fasterRCNN.create_architecture()

    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1) 

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand 
        thresh_obj = args.thresh_obj
        vis = args.vis

    
        total_tic = time.time()
         
        im_in = image # NOTE that image should be loaded with cv2 (so that it is in BGR)

        # bgr
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 

        # pdb.set_trace()
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show = np.copy(im)

        
        obj_dets, hand_dets = None, None
        for j in xrange(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)


            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()


        # convert to PIL
        im = im2show[:,:,::-1]
        image = Image.fromarray(im).convert("RGBA")
        width, height = image.size 

        
        if (obj_dets is not None) and (hand_dets is not None):
            img_obj_id = filter_object(obj_dets, hand_dets)
            obj_bboxes = []
            for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
                bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
                score = obj_dets[i, 4]
                if score > thresh_obj and i in img_obj_id:
                    object_bbox = bbox
                    obj_bboxes.append(object_bbox)
                else:
                    object_bbox = None
            if obj_bboxes != []:
                object_bbox = union_box(*obj_bboxes)
  
            hand_bboxes = []
            for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
                bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
                score = hand_dets[i, 4]
                lr = hand_dets[i, -1]
                state = hand_dets[i, 5]
                if score > thresh_hand:
                    hand_bbox = bbox
                    hand_bboxes.append(hand_bbox)
                else:
                    hand_bbox = None
            if hand_bboxes != []:
                hand_bbox = union_box(*hand_bboxes)

        else:
            print('Could not find any hands or objects in the image.')
            object_bbox, hand_bbox = None, None
        
        return object_bbox, hand_bbox

## this one is with lisa and gsam
# def hand_object_detector(image):
#     from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
#     from model.rpn.bbox_transform import clip_boxes
#     # from model.nms.nms_wrapper import nms
#     from model.roi_layers import nms
#     from model.rpn.bbox_transform import bbox_transform_inv
#     from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
#     from model.utils.blob import im_list_to_blob
#     from model.faster_rcnn.vgg16 import vgg16
#     from model.faster_rcnn.resnet import resnet
#     import pdb

#     try:
#         xrange          # Python 2
#     except NameError:
#         xrange = range  # Python 3

#     args = Config()
#     args.dataset = 'pascal_voc'
#     args.net = 'res101'
#     args.set_cfgs = None
#     args.cuda = True
#     args.class_agnostic = False
#     args.webcam_num = -1
#     args.parallel_type = 0
#     args.checksession = 1
#     args.checkepoch = 8
#     args.checkpoint = 89999
#     args.batch_size = 1
#     args.vis = False
#     args.thresh_hand = 0.5
#     args.thresh_obj = 0.5

#     if args.cfg_file is not None:
#         cfg_from_file(args.cfg_file)
#     if args.set_cfgs is not None:
#         cfg_from_list(args.set_cfgs)

#     cfg.USE_GPU_NMS = args.cuda
#     np.random.seed(cfg.RNG_SEED)

#     # load model
#     model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
#     if not os.path.exists(model_dir):
#         raise Exception('There is no input directory for loading network from ' + model_dir)
#     load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

#     pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
#     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

#     # initilize the network here.
#     fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
#     fasterRCNN.create_architecture()

#     if args.cuda > 0:
#         checkpoint = torch.load(load_name)
#     else:
#         checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
#     fasterRCNN.load_state_dict(checkpoint['model'])
#     if 'pooling_mode' in checkpoint.keys():
#         cfg.POOLING_MODE = checkpoint['pooling_mode']


#     # initilize the tensor holder here.
#     im_data = torch.FloatTensor(1)
#     im_info = torch.FloatTensor(1)
#     num_boxes = torch.LongTensor(1)
#     gt_boxes = torch.FloatTensor(1)
#     box_info = torch.FloatTensor(1) 

#     # ship to cuda
#     if args.cuda > 0:
#         im_data = im_data.cuda()
#         im_info = im_info.cuda()
#         num_boxes = num_boxes.cuda()
#         gt_boxes = gt_boxes.cuda()

#     with torch.no_grad():
#         if args.cuda > 0:
#             cfg.CUDA = True

#         if args.cuda > 0:
#             fasterRCNN.cuda()

#         fasterRCNN.eval()

#         start = time.time()
#         max_per_image = 100
#         thresh_hand = args.thresh_hand 
#         thresh_obj = args.thresh_obj
#         vis = args.vis

    
#         total_tic = time.time()
         
#         im_in = image # NOTE that image should be loaded with cv2 (so that it is in BGR)

#         # bgr
#         im = im_in

#         blobs, im_scales = _get_image_blob(im)
#         im_blob = blobs
#         im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

#         im_data_pt = torch.from_numpy(im_blob)
#         im_data_pt = im_data_pt.permute(0, 3, 1, 2)
#         im_info_pt = torch.from_numpy(im_info_np)

#         with torch.no_grad():
#             im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
#             im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
#             gt_boxes.resize_(1, 1, 5).zero_()
#             num_boxes.resize_(1).zero_()
#             box_info.resize_(1, 1, 5).zero_() 

#         # pdb.set_trace()
#         det_tic = time.time()

#         rois, cls_prob, bbox_pred, \
#         rpn_loss_cls, rpn_loss_box, \
#         RCNN_loss_cls, RCNN_loss_bbox, \
#         rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

#         scores = cls_prob.data
#         boxes = rois.data[:, :, 1:5]

#         # extact predicted params
#         contact_vector = loss_list[0][0] # hand contact state info
#         offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
#         lr_vector = loss_list[2][0].detach() # hand side info (left/right)

#         # get hand contact 
#         _, contact_indices = torch.max(contact_vector, 2)
#         contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

#         # get hand side 
#         lr = torch.sigmoid(lr_vector) > 0.5
#         lr = lr.squeeze(0).float()

#         if cfg.TEST.BBOX_REG:
#             # Apply bounding-box regression deltas
#             box_deltas = bbox_pred.data
#             if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
#             # Optionally normalize targets by a precomputed mean and stdev
#                 if args.class_agnostic:
#                     if args.cuda > 0:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     else:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
#                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

#                     box_deltas = box_deltas.view(1, -1, 4)
#                 else:
#                     if args.cuda > 0:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
#                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
#                     else:
#                         box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
#                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
#                     box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

#             pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
#             pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
#         else:
#             # Simply repeat the boxes, once for each class
#             pred_boxes = np.tile(boxes, (1, scores.shape[1]))

#         pred_boxes /= im_scales[0]

#         scores = scores.squeeze()
#         pred_boxes = pred_boxes.squeeze()
#         det_toc = time.time()
#         detect_time = det_toc - det_tic
#         misc_tic = time.time()
#         im2show = np.copy(im)

        
#         obj_dets, hand_dets = None, None
#         for j in xrange(1, len(pascal_classes)):
#             # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
#             if pascal_classes[j] == 'hand':
#                 inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
#             elif pascal_classes[j] == 'targetobject':
#                 inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)


#             # if there is det
#             if inds.numel() > 0:
#                 cls_scores = scores[:,j][inds]
#                 _, order = torch.sort(cls_scores, 0, True)
#                 if args.class_agnostic:
#                     cls_boxes = pred_boxes[inds, :]
#                 else:
#                     cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
#                 cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
#                 cls_dets = cls_dets[order]
#                 keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
#                 cls_dets = cls_dets[keep.view(-1).long()]
#                 if pascal_classes[j] == 'targetobject':
#                     obj_dets = cls_dets.cpu().numpy()
#                 if pascal_classes[j] == 'hand':
#                     hand_dets = cls_dets.cpu().numpy()


#         # convert to PIL
#         im = im2show[:,:,::-1]
#         image = Image.fromarray(im).convert("RGBA")
#         width, height = image.size 

#         object_bbox, hand_bbox = obj_hand_bboxes(im)
        
#         if (obj_dets is not None) and (hand_dets is not None):
#             img_obj_id = filter_object(obj_dets, hand_dets)
#             obj_bboxes = []
#             for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
#                 bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
#                 score = obj_dets[i, 4]
#                 if score > thresh_obj and i in img_obj_id:
#                     object_bbox = bbox
#                     obj_bboxes.append(object_bbox)
#                 else:
#                     object_bbox = None
#             if obj_bboxes != []:
#                 object_bbox = union_box(*obj_bboxes)
  
#             hand_bboxes = []
#             for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
#                 bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
#                 score = hand_dets[i, 4]
#                 lr = hand_dets[i, -1]
#                 state = hand_dets[i, 5]
#                 if score > thresh_hand:
#                     hand_bbox = bbox
#                     hand_bboxes.append(hand_bbox)
#                 else:
#                     hand_bbox = None
#             if hand_bboxes != []:
#                 hand_bbox = union_box(*hand_bboxes)

#         elif (obj_dets is None) and (hand_dets is not None):
#             # # object_bbox, hand_bbox = None, None
#             # obj_mask, object_bbox = obj_mask_and_bbox(im) 
#             # # can also return obj_mask if necessary

#             object_cc_list = []
#             object_cc_list.append(calculate_center(object_bbox))
#             object_cc_list = np.array(object_cc_list)
#             img_obj_id = []
#             for i in range(hand_dets.shape[0]):
#                 if hand_dets[i, 5] <= 0:
#                     img_obj_id.append(-1)
#                     continue
#                 hand_cc = np.array(calculate_center(hand_dets[i,:4]))
#                 point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
#                 dist = np.sum((object_cc_list - point_cc)**2,axis=1)
#                 dist_min = np.argmin(dist)
#                 img_obj_id.append(dist_min)
            
#             for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
#                 bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
#                 score = hand_dets[i, 4]
#                 lr = hand_dets[i, -1]
#                 state = hand_dets[i, 5]
#                 if score > thresh_hand and i in img_obj_id:
#                     hand_bbox = bbox
#             if img_obj_id[0] == -1: # last resort (this is because in some images of the test split, hand and object are not touching, so img_obj_id is -1)
#                 hand_bbox = bbox

#         elif (obj_dets is not None) and (hand_dets is None):
#             hand_mask, hand_bbox = hand_segmentation(im)
#             obj_mask, object_bbox = obj_mask_and_bbox(im) 

#         else:
#             print('Could not find any hands or objects in the image.')
#             object_bbox, hand_bbox = None, None
        
#         return object_bbox, hand_bbox
