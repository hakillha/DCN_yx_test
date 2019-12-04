from __future__ import absolute_import
import sys
sys.path.insert(0, '/media/yingges/Data/201910/Experiments/Test01/PG02/DCN/DCN_yx_test')

import DCN.fpn.inference as inference

image_names = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/images/+20190728-755HA-195610-1564813995814+20190728102826420+original_images+1390.jpg'
model_path = '/media/yingges/Data/201910/Deploy/DCN_GPU_cu100/DCN/model/epoch13/rfcn_dcn_voc'

print(inference.batch_img_infer([image_names],
								model_path,
								'fpn'))