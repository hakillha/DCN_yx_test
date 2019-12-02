# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yi Li, Haochen Zhang
# --------------------------------------------------------
import json
import logging
import os
import pprint
import sys

import cv2
import numpy as np
import mxnet as mx

from os.path import join as pj

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path)
import _init_paths
from utils.image import resize, transform
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from config.config import config, update_config
# update_config(cur_path + '/../experiments/rfcn/cfgs/resnet_v1_101_voc0712_rfcn_dcn_end2end_ohem.yaml')
update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml')
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

# PREDEFINED_CLASSES = ['i','p', 'wo', 'rn', 'lo', 'tl',  'ro']
PREDEFINED_CLASSES = ['io', 'wo', 'ors', 'p10', 'p11', 
                      'p26', 'p20', 'p23', 'p19', 'pne',
                      'rn', 'ps', 'p5', 'lo', 'tl',
                      'pg', 'sc1','sc0', 'ro', 'pn',
                      'po', 'pl', 'pm']

def res2jsondict(res, out_json_dict, im_name, img_id, anno_id):
    for bbox, score, cat_name, cat_id in zip(res['bbox'], res['score'], res['cat_name'], res['cat_id']):
        bb = [int(bb_) for bb_ in [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]]
        area = bb[2] * bb[3]
        anno_entry = {'area': area,
                      'bbox': bb,
                      'iscrowd': 0,
                      'image_id': img_id,
                      'category_name': cat_name,
                      'category_id': cat_id,
                      'id': anno_id,
                      'ignore': 0,
                      'score': float(score)}
        anno_id += 1
        out_json_dict['annotations'].append(anno_entry)
    if len(res['score']) > 0:
        img = cv2.imread(im_name)
        height, width, _ = img.shape
        img_info = {
            'file_name': os.path.basename(im_name),
            'height': height,
            'width': width,
            'id': img_id
        }
        out_json_dict['images'].append(img_info)
        img_id += 1
        return img_id, anno_id

def res2jsonlist(res, out_json_list, image_id_map):
    im_name = res['file_name']
    for bbox, score, cat_name, cat_id in zip(res['bbox'], res['score'], res['cat_name'], res['cat_id']):
        bb = [int(bb_) for bb_ in [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]]
        area = bb[2] * bb[3]
        anno_entry = {'area': area,
                      'bbox': bb,
                      'image_id': image_id_map[im_name],
                      'category_name': cat_name,
                      'category_id': cat_id,
                      'score': float(score)}
        out_json_list.append(anno_entry)

def dataset_img_infer(image_names,
                      model_path, 
                      json_file, 
                      image_info_list,
                      model_type,
                      cuda_provided,
                      display,
                      batch=64, 
                      viz_json_file=None, 
                      classes=None,
                      thresh=1e-3,
                      classes_map=None,
                      ):
    """
    Args:
        image_names: The list of test imgs names. Absolute path is strongly recommended.
        model_path: Path to the model file.
        json_file: The path of the json file that stores the output results.
        image_info_list: For ease of evaluation pls convert the input data into coco format
                      and pass in the 'images' list
        viz_json_file: Specify this when you want to store the output for visualization
        thresh: Confidence threshold
        classes_map: Specify this when class ids don't start sequentially from 1

    """
    # pathchecks here
    # model_path
    # json_file
    image_id_map = {}
    for info in image_info_list:
        image_id_map[info['file_name']] = info['id']

    if classes == None:
        classes = PREDEFINED_CLASSES
    if classes_map == None:
        classes_map = {}
        for idx, cls_ in enumerate(classes):
            classes_map[cls_] = idx + 1

    out_json_list = []

    if viz_json_file is not None:
        viz_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}
        img_id = 0
        anno_id = 0
        for cat, cid in classes_map.items():
            cat_entry = {'supercategory': 'none', 'id': cid, 'name': cat}
            viz_json_dict['categories'].append(cat_entry)
        viz_json_dict['categories'].sort(key=lambda val: val['id'])

    image_names =  [im_name for im_name in image_names if os.path.basename(im_name) in image_id_map.keys()]
    steps = int(len(image_names) / batch) + 1
    for step in range(steps):
        data = image_names[step * batch:] if step == steps - 1 else image_names[step * batch:(step + 1) * batch]
        res_list = batch_img_infer(data, model_path, model_type, classes=classes, thresh=thresh, cuda_provided=cuda_provided, display=display)
        for res, im_name in zip(res_list, data):
            # skip the image files that are not in the json ann file
            # if not os.path.basename(im_name) in image_id_map.keys():
            #     continue
            res2jsonlist(res, out_json_list, image_id_map)

            if viz_json_file is not None:
                ids = res2jsondict(res, viz_json_dict, im_name, img_id, anno_id)
                if ids is not None:
                    img_id, anno_id = ids

        print(str(step + 1) + '/' + str(steps))

    out_f = open(json_file, 'w')
    out_f.write(json.dumps(out_json_list))
    out_f.close()

    if viz_json_file is not None:
        out_f = open(viz_json_file, 'w')
        out_f.write(json.dumps(viz_json_dict))
        out_f.close()

def batch_img_infer(image_names,
                    model_path,
                    model_type,
                    output_coco_json=False, 
                    json_file=None, 
                    classes=None, 
                    thresh=1e-3, 
                    classes_map=None,
                    display=False,
                    cuda_provided=False):
    """

    Args:
        image_names: The list of test imgs names. Absolute path is strongly recommended.
                     This function can take one image at a time but you still need to 
                     enclose the name in a list.
        model_path: Path to the model file.
        output_coco_json: Set this to true when you want to store the output.
        json_file: The path of the json file that stores the output results.
        thresh: Confidence threshold

    Returns:
        A list of dictionaries each of which is the output for one image, formatted as:
            [{'file_name': File name of the input image,
             'bbox': A list of bounding boxes with following format [tlx, tly, brx, bry], 
             'score': A list of corresponding scores, 
             'cls_name': A list of corresponding class names as strings},
             {...},
             {...},
             ...]

    """
    tic()         
    # get symbol
    # pprint.pprint(config)
    if model_type == 'rfcn':
        config.symbol = 'resnet_v1_101_rfcn_dcn'
    elif model_type == 'fpn':
        config.symbol = 'resnet_v1_101_fpn_dcn_rcnn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    if classes == None:
        classes = PREDEFINED_CLASSES
    if classes_map == None:
        classes_map = {}
        for idx, cls_ in enumerate(classes):
            classes_map[cls_] = idx + 1
    num_classes = len(classes) + 1 # plus bg

    data = []
    for im_name in image_names:
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})
    data_names = ['data', 'im_info']

    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]

    arg_params, aux_params = load_param(model_path, 0, process=True)
    if cuda_provided:
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
    else:
        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.cpu()], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
    # nms = gpu_nms_wrapper(config.TEST.NMS, 0)
    nms = py_nms_wrapper(config.TEST.NMS)

    # warm up
    # for j in xrange(2):
    #     data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
    #                                  provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
    #                                  provide_label=[None])
    #     scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    #     scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)

    # test
    res_list = []
    if output_coco_json:
        out_json_dict = {'images': [], 'type': 'instances', 'annotations':[], 'categories':[]}
        img_id = 0
        anno_id = 0
        for cat, cid in classes_map.items():
            cat_entry = {'supercategory': 'none', 'id': cid, 'name': cat}
            out_json_dict['categories'].append(cat_entry)
        out_json_dict['categories'].sort(key=lambda val: val['id'])

    max_per_image = config.TEST.max_per_image
    all_boxes = [[] for _ in range(num_classes)]
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if config.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[j] = cls_dets[keep, :]

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        # visualize
        if display:
            im = cv2.imread(im_name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            show_boxes(im, all_boxes[1:], classes, 1)

        res = {'file_name': os.path.basename(im_name), 'bbox': [], 'score': [], 'cat_name': [], 'cat_id': []}
        for cls_idx, cls_name in enumerate(classes): # order of the iteration
            # this line might break down when the class ids don't grow sequentially from 1
            cat_id = cls_idx + 1 
            cls_dets = all_boxes[cat_id] # add one to ignore bg
            for det in cls_dets:
                res['bbox'].append(list(det)[:4])
                res['score'].append(det[-1])
                res['cat_name'].append(cls_name)
                res['cat_id'].append(cat_id)
        res_list.append(res)

        if output_coco_json:
            ids = res2jsondict(res, out_json_dict, im_name, img_id, anno_id)
            if ids is not None:
                img_id, anno_id = ids

        if display:
            print(res)

        print(toc())

    if output_coco_json:   
        out_f = open(json_file, 'w')
        out_f.write(json.dumps(out_json_dict))
        out_f.close()

    return res_list
