import os
from datetime import datetime
PROPOSAL_NUM = ''
CAT_NUM=''
if_classification = False 
BATCH_SIZE = 4
INPUT_SIZE = (224, 224)  # (w, h)
bigger = False

LR = 0.0005 # learning rate
WD = 1e-4   # weight decay 0.0001
SAVE_FREQ = 1
loss_weight_mask_thres = -1
pretrain = True#True

use_attribute = ['11','12','21','22']


test_model = 'model.ckpt'
model_name = 'resnext50_32x4d'  #濡傛灉鏄痠nception鐨勮瘽 INPUT_SIZE瑕佹敼鎴�299,299
model_size = '201'
loss_name = 'L2' #鏍规嵁train.py涓�鐨勮�惧畾淇�鏀癸紝鍦╰rain.py鐨�82琛屻€�"L1": 瀵归毦鏍锋湰涓嶆槸鐗瑰埆鏁忔劅锛�"L2": 瀵归毦鏍锋湰鏈€鏁忥紱"smooth_L1","huber": 瀵归毦鏍锋湰鏈€涓嶆晱鎰�,杩欎袱涓�闇€瑕佺‘瀹歭oss_alpha
loss_alpha = 1 #0.5,0.8,0.3

dataset_size = 435

flip_prob = 0.5#鏁版嵁鎵╁�� 缈昏浆姒傜巼
crop_method = 1  #0浠ｈ〃鏁村紶x鍏夎寖鍥�, 1浠ｈ〃涓婄墮榻胯寖鍥�

save_dir = './data\\model_save\\' #淇濆瓨妯″瀷鐨勮矾寰�
file_dir = './train_files\\'  #娴嬭瘯缁撴灉
time_first =  datetime.now().strftime('%Y%m%d_%H%M%S')

resume = "" # os.path.join(save_dir, '20210512_172826kfold_may5_revised_crop1_725_aug_p_0_attri_7_8_image_randomresnext101_32x8d_101pretrain-Falsesize224_0','model_param.pkl')
start_from_test_id = 1

#娴嬭瘯璺�寰�
load_time = '20210627_201942'
load_file = 'part1_jun11_revised_crop1_725_aug_p_0_attri_4_5resnext101_32x8d_101pretrain-Falsesize224_4'

load_model_path = os.path.join(save_dir, load_time+load_file,'model_param.pkl')
anno_csv_path = "male_may9_936_after_revise2_crop_1_435train_kfold.csv".format(crop_method, dataset_size)  #may9_936_after_revise2_crop_1_{}train_kfold.csv".format(dataset_size)#1_936_nov_18_725train_output.csv"
test_anno_csv_path = "male_may9_936_after_revise2_crop_1_435train_kfold.csv".format(dataset_size) #may9_936_after_revise2_crop_1_{}train_kfold.csv".format(dataset_size)


##鍙�鏀硅繖閲�
use_part =1 #(姣斿�俻art1-1)锛屽�瑰簲need_attributes_idx_total涓�鐨勭��(use_part+1)琛�
use_gpu = '0' #str(use_part%8) 閫氳繃nvidia-smi鍛戒护鏌ョ湅绌洪棽鐨刧pu缂栧彿锛屼竴涓�缂栧彿鏄�涓€寮犲崱锛屼笉瑕佷竴娆″崰涓ゅ紶鍗�
need_attributes_idx_total = [[4,5,6,7], #鐗欓娇瑙掑害鐨勬爣鍙穃
                              [4,5,6,7,8,9,10,11,12], #鍩洪�ㄥ�藉害锛岄€夋嫨鍦ㄥ�勭悊涔嬪悗鐨勮〃鏍间腑鐨勫垪鏁� 濡傝�掑害鍦ㄥ�勭悊鍚庣殑琛ㄦ牸涓�鏄�4,5,6鍒� 鍩洪�ㄥ湪澶勭悊鍚庣殑琛ㄦ牸涓�鏄�9,30,32\ smr涓�鏄�567鍒�
                              [8], #鍩洪�ㄩ暱搴�
                              [9,10,11],#宓撮《-鍩洪�ㄩ暱搴�
                              [9,10,11,12],\
                              [31,28,25],
                              [10,11],
                             [12],
                             [13]]
save_name = 'part{}attri_{}_{}'.format(use_part,need_attributes_idx_total[use_part][0],need_attributes_idx_total[use_part][-1])+ model_name+'_'+ model_size
test_save_name = 'test'+time_first+save_name  #'test'+load_time+load_file # 'kfold_may5_revised_crop1_{}_aug_p_{}_attri_{}_{}'.format(dataset_size, flip_prob,need_attributes_idx_total[0][0],need_attributes_idx_total[0][-1])+ model_name+'_'+ model_size+"pretrain-"+str(pretrain)
file_dir_test = 'test_files\\'+test_save_name

for i in range(len(need_attributes_idx_total)):
    for j in range(len(need_attributes_idx_total[i])):
        need_attributes_idx_total[i][j] -= 0
need_attributes_idx = need_attributes_idx_total[use_part]
max_epoch =700
use_uniform_mean = '12'

#濡傛灉涓嶆槸4涓�鐗欎綅涓€璧风畻鐨勮瘽锛寀se uniform瑕佺瓑浜巙se_attribute
