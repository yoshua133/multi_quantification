

#os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu


if __name__ ==  '__main__':
    import os
    import numpy as np
    import shutil
    import torch.utils.data
    from torch.nn import DataParallel
    from datetime import datetime
    from torch.optim.lr_scheduler import MultiStepLR
    from config_classification import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir, max_epoch, need_attributes_idx,use_uniform_mean,anno_csv_path, \
    use_gpu, save_name, model_size, pretrain,loss_weight_mask_thres, model_name, bigger, start_from_test_id, test_save_name,file_dir_test,time_first,loss_name,\
    loss_alpha, if_classification,num_of_need_attri
    
    import torch.nn.functional as F
    from core import  resnet
    from core import  dataset_class as dataset
    from core.utils import init_log, progress_bar
    import pandas as pd
    import torchvision.models  
    from IPython import embed
    import time
    
    
    start_epoch = 0
    num_of_need_attri = num_of_need_attri
    print("use attribute",need_attributes_idx)
    print("cuda available", torch.cuda.is_available())
    print("start training")



    save_dir_ori = save_dir
    file_dir_ori = file_dir
    #time_first =  datetime.now().strftime('%Y%m%d_%H%M%S')
    former_best = list()
    #for test_id in range(5):
    test_id = start_from_test_id
    #if start_from_test_id>test_id:
    #    continue
    if model_name == 'resnet':
        if model_size == '50':
            net = resnet.resnet50(pretrained=pretrain, num_classes = num_of_need_attri,bigger=bigger )           
        elif model_size == '34':
            net = resnet.resnet34(pretrained=pretrain, num_classes = num_of_need_attri )
        elif model_size == '101':
            net = resnet.resnet50(pretrained=pretrain, num_classes = num_of_need_attri,bigger=bigger )
        elif model_size == '152':
            net = resnet.resnet152(pretrained=pretrain, num_classes = num_of_need_attri )        
    elif model_name == 'vgg':
        if model_size == '11':
            net = torchvision.models.vgg11_bn(pretrained=pretrain, num_classes = num_of_need_attri )
        elif model_size == '16':
            net = torchvision.models.vgg16_bn(pretrained=pretrain, num_classes = num_of_need_attri )
        elif model_size == '16_nobn':
            net = torchvision.models.vgg16(pretrained=pretrain, num_classes = num_of_need_attri )
        elif model_size == '19':
            net = torchvision.models.vgg19_bn(pretrained=pretrain, num_classes = num_of_need_attri )
            
    elif model_name == "resnext101_32x8d":
        net = torchvision.models.resnext101_32x8d(pretrained=pretrain, num_classes = num_of_need_attri )
    
    elif model_name == "resnext50_32x4d":
        net = torchvision.models.resnext50_32x4d(pretrained=pretrain, num_classes = num_of_need_attri )
    
    elif model_name == "inception_v3":
        net = torchvision.models.inception_v3(pretrained=pretrain, num_classes = num_of_need_attri, aux_logits =False )
        
    elif model_name == "wide_resnet101_2":
        net = torchvision.models.wide_resnet101_2(pretrained=pretrain, num_classes = num_of_need_attri)
    elif  model_name == "wide_resnet50_2":
        net = torchvision.models.wide_resnet50_2(pretrained=pretrain, num_classes = num_of_need_attri)
        
    elif model_name == "densenet":
        if model_size == '121':
            net = torchvision.models.densenet121(pretrained=pretrain, num_classes = num_of_need_attri)             
        elif model_size == '161':
            net = torchvision.models.densenet161(pretrained=pretrain, num_classes = num_of_need_attri)  
        elif model_size == '169':
            net = torchvision.models.densenet169(pretrained=pretrain, num_classes = num_of_need_attri)  
        elif model_size == '201':
            net = torchvision.models.densenet201(pretrained=pretrain, num_classes = num_of_need_attri)        
        
        
    
    
    
        
    save_dir = os.path.join(save_dir_ori,time_first+save_name+"_{}".format(test_id))
    file_dir = os.path.join(file_dir_ori, time_first+save_name+"_{}".format(test_id))
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    if not os.path.exists(file_dir_test):
        os.makedirs(file_dir_test)
        
    if loss_name == "L1":   #对难样本不是特别敏感
        creterion = torch.nn.L1Loss(reduce= False)
    elif loss_name == "L2":  #对难样本最敏感
        creterion = torch.nn.MSELoss(reduce= False)
    elif loss_name == "smooth_L1":   #对难样本最不敏感
        creterion = torch.nn.SmoothL1Loss(reduce= False, beta = loss_alpha)
    elif loss_name == "huber":
        creterion = torch.nn.HuberLoss(reduce= False, delta = loss_alpha)   
        
    if if_classification:
        creterion = torch.nn.CrossEntropyLoss(reduce= False)   
    print("use loss",loss_name)
    # read dataset
    trainset = dataset.tooth_dataset_train(anno_path=anno_csv_path,test_id = test_id)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=1, drop_last=False, pin_memory= False)
    testset = dataset.tooth_dataset_test(anno_path=anno_csv_path,test_id = test_id)
    testset.attributes_mean = trainset.attributes_mean
    testset.attributes_std = trainset.attributes_std
    print("test mean",testset.attributes_mean)
    print("test std",testset.attributes_std)
    testloader = torch.utils.data.DataLoader(testset, pin_memory= False)
    # define model
    
    
        
    #embed()
    if resume :
        ckpt = torch.load(resume)
        for name in list(ckpt.keys()):
          ckpt[name.replace('module.','')] = ckpt[name]
          del ckpt[name]
        net.load_state_dict(ckpt)
        start_epoch = 0#ckpt['epoch'] + 1
    
    
    # define optimizers
    raw_parameters = list(net.parameters())
    
    
    raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    #lr_schedule = optim.lr_schedule.StepLR(raw_optimizer,
    schedulers = [MultiStepLR(raw_optimizer, milestones=[160, 200], gamma=0.1)]
    net = net.cuda()
    net = DataParallel(net)
    
    average_loss = [[111.1,111.1,111.1,100.0,100.0]]
    average_loss.extend(former_best)
    head=['train_loss_unit_degree','train_ori_loss_unit_std','test_loss','test_ori_loss','target_loss']
    
    
    
    
    
    test_head=['cur_use_attri','teeth_place']
    for pre_name in ['output_original','output_logits']:
        for attr_id in range(num_of_need_attri):
            test_head.append(pre_name+'_'+str(attr_id))
    test_head.append('target')
    test_head.append('max_index')
    #test_head.append('predict_right')
    print("test_head",test_head)
    #test_save_name =  'part6_dec4'#str(datetime.now().strftime('%Y%m%d_%H%M%S')) 
    
    save_csv_path_test = os.path.join(file_dir,'test_dataset_{}.csv'.format(test_id))#,test_save_name))
    #save_csv_path_train = file_dir_test+'/{}_train_dataset_{}.csv'.format(test_save_name,test_id)
    
    
    
    
    
    
    
    
    for epoch in range(start_epoch, max_epoch):
       
    
        # begin training
        print('--' * 50)
        net.train()
        train_num = 0
        train_loss = 0
        train_ori_loss = 0
        
        
        print("before train")
        for i, data in enumerate(trainloader):
            if i%50==0:
                print("in train",i)
            img, target = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            # print("batch size",batch_size)
            train_num += batch_size
            raw_optimizer.zero_grad()
            output = net(img)
            print(output.shape, target.shape)
            #embed()
            target = target.reshape(-1)
            loss_delta = creterion(output,target)           
            weight = torch.ones(loss_delta.shape).cuda()
            #weight[loss< torch.tensor(loss_weight_mask_thres).cuda()] = 0.5  # this is the right way
            #weight[loss> torch.tensor(loss_weight_mask_thres/trainset.attributes_std[use_uniform_mean]).cuda().reshape(-1)] = 0.5   #loss is the delta for normed attribute, nothing to do with std
            #print("loss",loss)
            #print("weight",weight)
            loss = loss_delta * weight
            loss = loss.sum()
            
            score, max_index = torch.max(output,1)
            right_num = torch.sum(max_index.reshape(-1) == target.reshape(-1))
            precision = right_num/batch_size           
    
            train_ori_loss +=  right_num.cpu().detach().numpy()
            train_loss += right_num.cpu().detach().numpy()
            loss.backward()
            raw_optimizer.step()
            #progress_bar(i, len(trainloader), 'train')
        for scheduler in schedulers:
            scheduler.step()
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        if not os.path.exists(file_dir):
                os.mkdir(file_dir)
        if epoch<1  :
            shutil.copy( 'config.py', file_dir+'/config.py')
            shutil.copy( 'train.py', file_dir+'/train.py')
            shutil.copy( 'core/dataset.py', file_dir+'/dataset.py')
            shutil.copy( 'core/resnet.py', file_dir+'/resnet.py')
        if epoch % 5 == 0 or epoch==1:
            test_loss = 0
            test_ori_loss = 0
            test_target_loss = 0
            test_num = 0
            net.eval()
            
            total_time = 0
            output_csv = []
            mae_total = 0
            mse_total = 0
            precision_total = 0
            seg_dict = {1:0,2:0,5:0,10:0}
            
            for i, data in enumerate(testloader):
                with torch.no_grad():
                    img, target = data[0].cuda(), data[1].cuda()
                    cur_use_attri, index = data[2],data[3]
                    
                    batch_size = img.size(0)
                    #print('test batch size',batch_size)#bs=1
                    test_num += batch_size
                    raw_optimizer.zero_grad()
                    start = time.time()
                    output = net(img)
                    end = time.time()
                    total_time += (end-start)
                    # calculate loss
                    #print("target",target.shape)
                    #print("target type",type(target))
                    #print("outputs",output.shape)
                    #print("output type",type(output))
                    #print("loss",loss)
                    #loss = creterion(output, target)
                    
                    logits = F.softmax(output,dim =1).reshape(-1)  #logit = Batch * num_class
                    score, max_index = torch.max(output,1)
                    right_num = torch.sum(max_index.reshape(-1) == target.reshape(-1))
                    precision = right_num/batch_size
                    precision_total += right_num
                    
                    cur_row =[]
                    cur_row.append(str(cur_use_attri[0]))#.item()))                                                                                                 
                    cur_row.append(str(index))
                    
                    for tar in output.reshape(-1):
                        #print('t',tar)
                        cur_row.append(str(tar.cpu().detach().numpy()))
                    for out in logits.reshape(-1) :
                        cur_row.append(str(out.cpu().detach().numpy()))
                    cur_row.append(str(target.cpu().detach().numpy()))
                    cur_row.append(str(max_index.cpu().detach().numpy()))
                    #right_cur = int(max_index == target)
                    
                    #cur_row.append(str(right_cur))
                        
                        
                    
                    
                    output_csv.append(cur_row)
                    
                    #loss is the mean distance between two tensor
                    test_loss += right_num.cpu().detach().numpy()
                    test_ori_loss += right_num.cpu().detach().numpy()
                    # calculate accuracy
                    
            output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
            mae_print = list()
          
            mae_print.append(str(precision_total/test_num))
            
            
            output_csv.insert(0,mae_print)
            output_csv.insert(0,["0~1",str(seg_dict[1]/test_num)])
            output_csv.insert(0,["1~2.5",str(seg_dict[2]/test_num)])
            output_csv.insert(0,["2.5~5",str(seg_dict[5]/test_num)])
            output_csv.insert(0,["5~10",str(seg_dict[10]/test_num)])
            #embed()
            loss_csv=pd.DataFrame(columns=test_head,data=output_csv)
            #embed()
            loss_csv.to_csv(save_csv_path_test,encoding='gbk')
    
    
    
    
    
            print("epoch:{} mean loss, L1 gap divided by std".format(epoch),test_loss/test_num,"  ori loss ",\
              test_ori_loss/test_num,"target loss", test_target_loss/test_num)
            print("test_num",test_num)
            #train_ori_loss = trainset.attributes_std[use_uniform_mean][0]*train_loss.item()/train_num
            #test_ori_loss = trainset.attributes_std[use_uniform_mean][0]*test_loss.item()/test_num
            average_loss.append([train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num, test_target_loss/test_num])
            if test_target_loss/test_num < average_loss[0][4]:
                average_loss[0] = [train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num, test_target_loss/test_num]
            loss_csv=pd.DataFrame(columns=head,data=average_loss)
            loss_csv.to_csv(file_dir+'/{}_loss.csv'.format(save_name),encoding='gbk')
            f = open(file_dir+'/{}_mean.txt'.format(save_name),'w')
            f.write(str(trainset.attributes_mean))
            f.close()
            f2 = open(file_dir+'/{}_std.txt'.format(save_name),'w')
            f2.write(str(trainset.attributes_std))
            f2.close()
            print("finish writing")
            net_state_dict = net.state_dict()
            torch.save(net_state_dict,save_dir+'/model_param.pkl')
            print("finish save")
    li = list()
    li.append(average_loss[0])
    former_best.extend(li)