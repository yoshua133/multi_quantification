import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv
from IPython import embed
csv_list = list()
train_loss_total = list()
test_loss_total = list()
train_l1_gap_total = []
test_l1_gap_total = []





#csv_list.append("/data/shimr/teethcode_2021_jan30/train_files/20210618_140724kfold_may5_revised_crop0_725_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_4/kfold_may5_revised_crop0_725_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_loss.csv")

#csv_list.append("/data/shimr/teethcode_2021_jan30/train_files/20210619_110200kfold_may5_revised_crop0_580_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_4/kfold_may5_revised_crop0_580_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_loss.csv")

#csv_list.append("/data/shimr/teethcode_2021_jan30/train_files/20210618_140849kfold_may5_revised_crop0_435_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_4/kfold_may5_revised_crop0_435_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_loss.csv")

csv_list.append("train_files\\20231121_191302malepart1attri_4_12resnext50_32x4d_201_1\\part1attri_4_12resnext50_32x4d_201_loss.csv")#/data/shimr/teethcode_2021_jan30/train_files/20210619_110106kfold_may5_revised_crop0_290_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_4/kfold_may5_revised_crop0_290_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_loss.csv")

sizes = [725*4,580*4,435*4,290*4]
colors = ['red','blue','green','orange']
for i in range(len(csv_list)):
    num = 0 
    train_loss = list()
    test_loss = list()
    train_l1_gap = []
    test_l1_gap = []
    csv_path = csv_list[i]
    with open(csv_path) as f:   
          for row in f:
              num +=1
              if num<3 or num>245:  #150
                continue   
                
              #embed()
              row = row.strip('\n').split(',')
              train_loss.append(float(row[2]))
              test_loss.append(float(row[4]))
              train_l1_gap.append(float(row[1]))
              test_l1_gap.append(float(row[3]))
              
    train_loss_total.append(train_loss)
    test_loss_total.append(test_loss)
    train_l1_gap_total.append(train_l1_gap)
    test_l1_gap_total.append(test_l1_gap)
fig = plt.figure(0)
#embed()
for i in range(len(csv_list)):
    print(i)
    size= sizes[i]
    color = colors[i]
    #embed()
    test_l1_gap = test_l1_gap_total[i]
    train_l1_gap = train_l1_gap_total[i]
    inter = 1
    test_l1_gap_inter = test_l1_gap[0:len(test_l1_gap):inter]
    x1 = range(0, 5*len(train_l1_gap),5)
    x2 = range(0, 5*len(test_l1_gap_inter)*inter,5*inter)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x1, train_l1_gap, '--',lw= 1,color = color)#, label="Error Train {}".format(size))
    plt.plot(x2, test_l1_gap_inter, '.-',lw= 1,markersize=0.6 ,color = color, label="size of {}".format(size))
    
plt.annotate('dashed: train', xy =(900, 0.8),
                xytext =(930, 1.3), 
                arrowprops = dict(arrowstyle='->'),)
plt.annotate('solid: test/valid', xy =(900, 2.7),
                xytext =(930, 3.2), 
                arrowprops = dict(arrowstyle='->'),)
plt.title('Prediction error over training epoches')
plt.ylabel('Error')
plt.legend(loc='best')
plt.savefig('./train_test_l1_gap.tif')
plt. close(0)

fig = plt.figure(0)
for i in range(len(csv_list)):
    size= sizes[i]
    train_loss = train_l1_gap_total[i]
    test_loss = test_l1_gap_total[i]
    plt.plot(x1, train_loss, '.-',label="Train Loss")# {}".format(size))
    plt.plot(x1, test_loss, '.-',label="Validation Loss")# {}".format(size))
plt.rc('font',family='Times New Roman')
plt.title('Loss over training epoches', fontsize= 'xx-large', family='Times New Roman')
plt.ylabel('Loss', fontsize ='xx-large', family='Times New Roman')
plt.xlabel('Epoch', fontsize ='xx-large', family='Times New Roman')
plt.legend(loc='upper right',fontsize = 'xx-large')
plt.savefig('./pretrain=trueflip=1resnext50.tif')
      
        