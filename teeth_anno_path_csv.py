# coding=utf-8
import os
import pdb
from PIL import Image
import glob
import pandas as pd
from pandas import DataFrame
import csv
import cv2
import random
import numpy as np
import os
from IPython import embed
from config import  crop_method
pation = 0.6 #训练数据集占总数据集的多少分之一，剩下的部分是验证集和测试集  0.8 0.6,0.4
test_pation = 0.2 #测试集占总数据集的多少，
images_dir = 'E:\\ai_images_2_jigu1-1000(important)\\'
anno_path = 'jigu4femaletooth1-1000 .csv'
#
output_path = 'female_after_revise2_crop_{}_{}train_kfold.csv'.format(crop_method,435)#int(725*pation))

teeth_row = 3 #必须昿 因为dataset.py中读数据的row是固定的  实际上是第四列
patient_row = 0

map_of_patient_id_kfold = dict()
reference_csv = ''#'may9_936_after_revise2_crop_1_725train_kfold_for_reference.csv'
if reference_csv:
    ref_num = 0
    ref = csv.reader(open(reference_csv, encoding='utf-8'))
    for line in ref:
        ref_num+=1
        patiend_id = line[0]
        kfold_id = line[2]
        map_of_patient_id_kfold[patiend_id] = kfold_id
#embed()
def crop_image(in_path,out_path):
  #print(in_path)
  img = cv2.imread(in_path)
  #print(img.shape)
  if crop_method==1:
        cropped = img[310:497,719:1023,:]#[310:497,719:919,:]#[310:497,719:1023,:]#[100:707,673:1280,:] 调了crop的范围之后，需要修改prefix，以便区分
  elif crop_method==0:
        cropped = img[100:707,673:1280,:]
  elif crop_method==2:
        cropped = img[100:707,673:1280,:]
  #cropped = img[50:727,623:1280,:]
  cv2.imwrite(out_path, cropped)

np.random.seed(22) #22

csv_num=0
r = csv.reader(open(anno_path, encoding='utf-8'))
#lines = [l for l in r]
#print(lines)

output_csv = []
former_id = -1
last_flag = 'unknown'
if crop_method == 0:
    prefix = "cropped_image_a"
elif crop_method == 1:
    prefix = "cropped_image"
else:
    prefix = "cropped_image_"+str(crop_method)+"_"
    
for line in r:
    csv_num+=1
    if csv_num ==0 :#or csv_num<2800:
        continue
    patient_id = str(line[patient_row]).zfill(3)
    tooth_id = str(line[teeth_row])
    #if int(patient_id)%1==0:
    print("patient_id",patient_id)
    #print("tooth_id",tooth_id)
    patient_id_path = os.path.join(images_dir,patient_id)
    if not os.path.exists(patient_id_path):
        continue
    teeth_files = os.listdir(patient_id_path)
    existed = False
    #print("patient_id_path",patient_id_path)
    for dir in teeth_files:    #if tooth_id in dir[4:7] and dir.endswith('tif') and (not 'crop' in dir):
         #print("dir",dir)
         
         if not ',' in dir:
            #print("no ,")
            continue
         if tooth_id in dir.split(',')[1] and 'tif' in dir and (not 'crop' in dir):
            existed = True
            image_file = dir
            tooth_tif_path = os.path.join(patient_id_path, dir)
    if existed:
        cropped_path = os.path.join(patient_id_path, prefix+image_file)
        #crop_image(tooth_tif_path,cropped_path)
        line[1] =  cropped_path
        ran = np.random.rand(1)
        ran2 = np.random.rand(1)
        if former_id == patient_id:
            line[2] = last_flag
            former_id = patient_id
        else:
            line[2] = int(ran/0.2)
            if ran < pation:
                line[2] = 'train'
            elif ran < pation+test_pation:
                line[2] = 'test'
            else:
                line[2] = 'val'
            #if line[0] in map_of_patient_id_kfold.keys():
            #    line[2] = map_of_patient_id_kfold[line[0]]
            #elif reference_csv:
            #    line[2] = ''
            last_flag = line[2]
            former_id = patient_id
        """
        elif ran <= 0.4:
          line[2] = 'train'
          last_flag = 'train'
          former_id = patient_id
        elif ran > 0.8:
          line[2] = 'test'
          last_flag = 'test'
          former_id = patient_id
        else:
          line[2] = 'val'
          last_flag = 'val'
          former_id = patient_id
        """
        try:
            crop_image(tooth_tif_path,cropped_path)
        except Exception:
            print(cropped_path,"has problem")
        else:
            output_csv.append(line)
        #print("line",line)
    else:
        print("patient_id",patient_id)
        print("fail")
    #embed()
writer = csv.writer(open(output_path,'w',newline=''))
writer.writerows(output_csv)
print("finish writing csv")