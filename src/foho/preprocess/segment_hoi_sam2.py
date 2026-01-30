import os
import sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from kiui.vis import plot_image as plti
from segment_anything import sam_model_registry, SamPredictor

from foho.configs import third_party_root

_TP = third_party_root()
sys.path.append(_TP)
sys.path.append(os.path.join(_TP, "estimator"))
from estimator.hand_object_detector.hoi_detector import hand_object_detector
sys.path.append(_TP)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  get affine transform matrix
    ---------
    @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
    -------
    @Returns: affine transformation matrix
    -------
    """

    def rotate_2d(pt_2d, rot_rad):
        x = pt_2d[0]
        y = pt_2d[1]
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        xx = x * cs - y * sn
        yy = x * sn + y * cs
        return np.array([xx, yy], dtype=np.float32)

    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_patch_image(cvimg, bbox, input_shape, do_flip, scale, rot):
    """
    @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                  generate the patch image from the bounding box and other parameters.
    ---------
    @param: input image, bbox(x1, y1, w, h), dest image shape, do_flip, scale factor, rotation degrees.
    -------
    @Returns: processed image, affine_transform matrix to get the processed image.
    -------
    """

    img = cvimg.copy()
    img_height, img_width, _ = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    new_trans = np.zeros((3, 3), dtype=np.float32)
    new_trans[:2, :] = trans
    new_trans[2, 2] = 1

    return img_patch, new_trans


def process_bbox(bbox, factor=1.25):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = 1.
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * factor
    bbox[3] = h * factor
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height
    # Compute areas of each box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute union
    union = area_box1 + area_box2 - intersection
    # Compute IoU
    return intersection / union if union > 0 else 0.0


def hoi_detector(img_path, hand_detector, sam_model, IoU_threshold, hand_object_detector, object_name=None):
    only_upto_wrist = False
    img_cv2 = cv2.imread(img_path)
    img_pil = Image.fromarray(img_cv2[..., ::-1])

    object_bbox, hand_bbox = hand_object_detector(img_cv2)
    bbox_obj = object_bbox.reshape((-1, 2))

    detections = hand_detector(img_cv2, conf=0.3, verbose=False, iou=IoU_threshold)[0] # conf=0.3

    bboxes = []
    is_right = []
    for det in detections: 
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())

    if len(bboxes) == 0:
        print("no hands in this image")
    elif len(bboxes) == 1:
        bbox_hand = np.array(bboxes[0]).reshape((-1, 2))
    elif len(bboxes) > 1:
        hand_idx = None
        max_iou = -10.
        for cur_idx, cur_bbox in enumerate(bboxes):
            cur_iou = calculate_iou(cur_bbox, bbox_obj.reshape(-1).tolist())
            if cur_iou >= max_iou:
                hand_idx = cur_idx
                max_iou = cur_iou
        bbox_hand = np.array(bboxes[hand_idx]).reshape((-1, 2))
        bboxes = [bboxes[hand_idx]]
        is_right = [is_right[hand_idx]]

    tl = np.min(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    br = np.max(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    box_size = br - tl
    bbox = np.concatenate([tl - 10, box_size + 20], axis=0)
    ho_bbox = process_bbox(bbox)
            
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    if not right:
        new_x1 = img_cv2.shape[1] - boxes[0][2]
        new_x2 = img_cv2.shape[1] - boxes[0][0]
        boxes[0][0] = new_x1
        boxes[0][2] = new_x2
        ho_bbox[0] = img_cv2.shape[1] - (ho_bbox[0] + ho_bbox[2])
        img_cv2 = cv2.flip(img_cv2, 1)
        right[0] = 1.

    scale_size = 512 # 224
    crop_img_cv2, trans = generate_patch_image(img_cv2, ho_bbox, (scale_size, scale_size), 0, 1.0, 0)
    crop_img_hoi = crop_img_cv2[..., ::-1]

    if object_name is None:
        object_name = "manipulated object"
    pred_obj = sam_model.predict([Image.fromarray(crop_img_hoi)], [object_name])
    if pred_obj[0]["boxes"] is None or len(pred_obj[0]["boxes"]) == 0:
        return None
    bbox_obj = pred_obj[0]["boxes"][0].reshape((-1, 2))
    mask_obj = pred_obj[0]["masks"][0]
    mask_obj = (mask_obj > 0).astype(np.uint8) # make the mask binary
    
    pred_hand = sam_model.predict([Image.fromarray(crop_img_hoi)], ["only hand"])
    if pred_hand[0]["boxes"] is None or len(pred_hand[0]["boxes"]) == 0:
        return None
    bbox_hand = pred_hand[0]["boxes"][0].reshape((-1, 2))
    mask_hand = pred_hand[0]["masks"][0]
    mask_hand = (mask_hand > 0).astype(np.uint8) # make the mask binary
    return mask_obj, mask_hand, crop_img_hoi, int(is_right[0])


def get_hoi_mask(source_image_path, hand_detector, sam_model, hand_object_detector, object_name='manipulated object'):
    """
    Given a source image, returns the hand and object masks. 
    """
    device = 'cuda'
    IoU_threshold = 0.5

    hand_detector = hand_detector.to(device)

    result = hoi_detector(source_image_path, hand_detector, sam_model, IoU_threshold, hand_object_detector, object_name)
    if result is None:
        return None
    crop_mask_obj, crop_hand_mask, crop_img_hoi, is_right = result
    
    # Create a mask for the whole object-hand interaction
    mask_hoi = np.logical_or(crop_mask_obj, crop_hand_mask)
    mask_hoi_3ch = mask_hoi[..., None].astype(bool)
    crop_mask_obj_3ch = crop_mask_obj[..., None].astype(bool)

    # Apply masks
    cropped_hoi_image_wo_bckg = crop_img_hoi * mask_hoi_3ch
    cropped_occ_obj_img = cropped_hoi_image_wo_bckg * crop_mask_obj_3ch

    # Set background (where mask is False) to white [255, 255, 255]
    background_mask = ~mask_hoi_3ch
    cropped_hoi_image_wo_bckg[background_mask.repeat(3, axis=2)] = 255

    cropped_occ_obj_img = cropped_hoi_image_wo_bckg * crop_mask_obj_3ch
    cropped_occ_obj_img[(~mask_hoi_3ch | ~crop_mask_obj_3ch).repeat(3, axis=2)] = 255

    torch.cuda.empty_cache()
    
    return cropped_occ_obj_img, crop_mask_obj, crop_hand_mask, cropped_hoi_image_wo_bckg, crop_img_hoi, is_right


if __name__ == "__main__":
    img_path = sys.argv[1]
    device = 'cuda'
    IoU_threshold = 0.5

    wilor_ckpt = os.environ.get(
        "WILOR_CKPT",
        os.path.join(third_party_root(), "estimator", "wilor_ckpt", "detector.pt"),
    )
    hand_detector = YOLO(wilor_ckpt)
    sam_model = LangSAM(sam_type="sam2.1_hiera_large")

    hand_detector = hand_detector.to(device)
    img_cv2 = cv2.imread(img_path)
    img_pil = Image.fromarray(img_cv2[..., ::-1])

    pred_obj = sam_model.predict([img_pil], ["manipulated object"])
    bbox_obj = pred_obj[0]["boxes"][0].reshape((-1, 2))
    mask_obj = pred_obj[0]["masks"][0]

    detections = hand_detector(img_cv2, conf=0.3, verbose=False, iou=IoU_threshold)[0]

    bboxes = []
    is_right = []
    for det in detections: 
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())

    if len(bboxes) == 0:
        print("no hands in this image")
    elif len(bboxes) == 1:
        bbox_hand = np.array(bboxes[0]).reshape((-1, 2))
    elif len(bboxes) > 1:
        hand_idx = None
        max_iou = -10.
        for cur_idx, cur_bbox in enumerate(bboxes):
            cur_iou = calculate_iou(cur_bbox, bbox_obj.reshape(-1).tolist())
            if cur_iou >= max_iou:
                hand_idx = cur_idx
                max_iou = cur_iou
        bbox_hand = np.array(bboxes[hand_idx]).reshape((-1, 2))
        bboxes = [bboxes[hand_idx]]
        is_right = [is_right[hand_idx]]

    tl = np.min(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    br = np.max(np.concatenate([bbox_obj, bbox_hand], axis=0), axis=0)
    box_size = br - tl
    bbox = np.concatenate([tl - 10, box_size + 20], axis=0)
    ho_bbox = process_bbox(bbox)
            
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    if not right:
        new_x1 = img_cv2.shape[1] - boxes[0][2]
        new_x2 = img_cv2.shape[1] - boxes[0][0]
        boxes[0][0] = new_x1
        boxes[0][2] = new_x2
        ho_bbox[0] = img_cv2.shape[1] - (ho_bbox[0] + ho_bbox[2])
        img_cv2 = cv2.flip(img_cv2, 1)
        right[0] = 1.

    crop_img_cv2, _ = generate_patch_image(img_cv2, ho_bbox, (224, 224), 0, 1.0, 0)
    crop_img_pil = Image.fromarray(crop_img_cv2[..., ::-1])

    pred_hand = sam_model.predict([Image.fromarray(crop_img_cv2[..., ::-1])], ["hand"])
    hand_mask = pred_hand[0]["masks"][0]

    crop_mask_obj, _ = generate_patch_image(mask_obj[..., None], ho_bbox, (224, 224), 0, 1.0, 0)
    crop_mask_obj = (crop_mask_obj > 0).astype(np.uint8) # make the mask binary
