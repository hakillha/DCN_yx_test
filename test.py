from __future__ import absolute_import

import json
import os
import random
import shutil
import sys

from os.path import join as pj

# from pycocotools.coco import COCO
#
# def extract_subset_coco(annFile, size=100):
# 	coco = COCO(annFile)
# 	imgIds = coco.getImgIds()
# 	random.shuffle(imgIds)
# 	imgs = coco.loadImgs(imgIds[:100])
# 	return [img['file_name'] for img in imgs]

scriptpath = "/media/yingges/Data/201910/Deploy/DCN_GPU_cu100/DCN"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

# batch_data_source = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/images'
batch_data_source = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/images'
test_img_path = [pj(batch_data_source, file) for file in os.listdir(batch_data_source) if file.endswith('.jpg')]
# gt_file = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/cocoformat_valid_out.json'
gt_file = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/valid.json'
gt_json = json.load(open(gt_file))

# model_path = '/media/yingges/Data/201910/Deploy/Deformable-ConvNets-CPU/DCN/model/rfcn_dcn_voc'
model_path = 'DCN/model/rfcn_dcn_voc'

# from rfcn import inference
import DCN.fpn.inference as inference

# extract_subset('/media/yingges/Data/201910/FT/FTData/ft_od1_merged',
# 			   '/media/yingges/Data/201910/FT/FTData/ft_od1_merged/sample1', 1)

inference.dataset_img_infer(test_img_path, model_path,
							'/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/epoch21_output.json',
							gt_json['images'], 'fpn', True, False,
							32)
