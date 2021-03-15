#This code is an adapted version of the original available here: https://github.com/MichiganCOG/TASED-Net
import os
import torch
import time
import numpy as np
import pandas as pd
import sys
import cv2
import json
import random
from datetime import timedelta
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchsummary import summary

from models.HD2S import HD2S as modelName
from loss import KLDLoss1vs1
from dataset.videoDataset import Dataset3D
from dataset.infiniteDataLoader import InfiniteDataLoader


def main():
    
    dev_name = 'cuda:0'
    
    pile = 25
    batch_size = 16
    len_temporal = 16
    validation_frac = 0.5 
    
    image_size=(128,192)
    
    num_iters = 5000
    num_workers = 2
    lr=0.001
    
    num_val_iter = 100
    encoder_pretrained= True
    file_weight='none'
    
    test_name= 'HD2S_training_demo_1'
    subfolder='BaseModel'
    path_source_data = [os.path.join('data','DHF1K','train')]
    
    source_loader=[None]*len(path_source_data)
    
    #validation on DHF1K
    path_validation = os.path.join('data','DHF1K','train')
    path_val_split = os.path.join(path_validation, 'splitTrainVal','valSet2.csv')
    valSet = pd.read_csv(path_val_split, dtype = str)['0'].values.tolist()
    
    path_output = os.path.join('output','model_weights',subfolder, test_name)
    
    for idx, p in enumerate(path_source_data):
        if 'LEDOV' in p:
            source_loader[idx] = InfiniteDataLoader(Dataset3D(p ,len_temporal, size=image_size), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            path_train_split = os.path.join(p,'splitTrainVal','trainSet2.csv')
            trainSet = pd.read_csv(path_train_split, dtype = str)['0'].values.tolist()
            source_loader[idx] = InfiniteDataLoader(Dataset3D(p ,len_temporal, size=image_size, list_videoName = trainSet), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print(test_name)
    
    dev = torch.device(dev_name if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(os.path.join('output', subfolder, test_name)):
        os.makedirs(os.path.join('output', subfolder, test_name))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)    

    '''
    # loading weights file (fine-tuning)
    weight_folder='HDS_training_demo_1'
    weight_name='weights_MinLoss.pt'
    file_weight=os.path.join('output','model_weights',subfolder,weight_folder,weight_name)
    
    optim_name='adam_MinLoss.pt'
    file_optimizer=os.path.join('output','model_weights',subfolder,weight_folder,optim_name)
    '''
    
    model = modelName(pretrained=encoder_pretrained)
    model = model.to(dev)
    '''
    # loading file weight (fine-tuning)
    model.load_state_dict(torch.load(file_weight, map_location=dev))
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-7)
    
    '''
    # loading optimizer (fine-tuning)
    #optimizer.load_state_dict(torch.load(file_optimizer))
    '''
   
    torch.backends.cudnn.benchmark = True
    criterion = KLDLoss1vs1(dev)
    
    model.train()
    
    #saving training info
    summ = summary(model, input_data=(3, 16, 128, 192), device= dev, verbose=0)
    
    info=['model_name: ', model.__class__.__name__ ,'\n',
          'path_source_data_train: ', str(path_source_data), '\n',
          'path_validation: ', str(path_validation), '\n',
          'pile: ', str(pile),'\n',
          'batch_size: ', str(batch_size),'\n',
          'len_temporal: ', str(len_temporal),'\n',
          'image_size: ',  str(image_size),'\n',
          'num_iters: ',  str(num_iters),'\n',
          'num_workers: ',  str(num_workers),'\n',
          'lr: ', str(lr), '\n',
          'num_val_iter: ', str(num_val_iter), '\n',
          'encoder_pretrained: ', str(encoder_pretrained), '\n',
          'file_weight: ', str(file_weight),'\n',
          'model_summary: ','\n', str(summ), '\n']
    
    file_info=open(os.path.join("output", subfolder, test_name, "train.txt"), 'w', encoding='utf-8')
    file_info.writelines(info)
    file_info.close()
    
    start_time = time.time()
    
    check_point={ 'step' : 0,
                 'MIN_loss_val' : sys.float_info.max,
                 'step_MIN_loss' : 0,
                 'exec_time' : 0,
                 'loss_history' : {'train_sal1':[], 'train_sal2':[], 'train_sal3':[], 'train_sal4':[], 'train_out':[] ,'train':[], 'validation':[]}
        }
    
    
    '''
    with open(os.path.join('output', subfolder, test_name, 'check_point.json')) as fp:
        check_point=json.load(fp)
    step = check_point['step']
    '''
        
    '''
    preparation dict for validation set
    '''
    valSet_list = []
    for v in tqdm(valSet, desc= "Preparing validation set"):
        
        path_frames= os.path.join(path_validation, 'frames', v)
        if 'DHF1K' in path_source_data:
            path_annt = os.path.join(path_validation, 'annotation', "%04d"% int(v), 'maps')
        else:    
            path_annt = os.path.join(path_validation, 'annotation', v, 'maps')
        list_frame_names = [f for f in os.listdir(path_frames) if os.path.isfile(os.path.join(path_frames, f))]
        list_annt_names = [a for a in os.listdir(path_annt) if os.path.isfile(os.path.join(path_annt, a))]
        list_frames = []
        list_annt = []
       
        for f in list_frame_names:
            img = cv2.imread(os.path.join(path_frames, f))
            img= cv2.resize(img, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
            list_frames.append(img)
            
        for a in list_annt_names:
            img = cv2.imread(os.path.join(path_annt, a), 0)
            img= cv2.resize(img, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
            list_annt.append(torch.from_numpy(img.copy()).contiguous().float())
        
        valSet_list.append({
            'name':v,
            'list_frame_names':list_frame_names,
            'list_annt_names':list_annt_names,
            'list_frames':list_frames,
            'list_annts':list_annt,
            'length': len(list_frame_names),
            }
        )
    
    num_videos_val = int(len(valSet)*validation_frac)
    
    #model losses
    loss_sum = 0
    loss1_sum = 0 
    loss2_sum = 0
    loss3_sum = 0
    loss4_sum = 0 
    lossOut_sum = 0
    
    index_sourceloader = -1
    
    for i in tqdm(range(check_point['step']*pile,num_iters*pile)):
        
        index_sourceloader = (index_sourceloader+1) % (len(path_source_data))
        
        clip_s, annt_s = next(source_loader[index_sourceloader])
        
        with torch.set_grad_enabled(True):
            clip_s = clip_s.to(dev)
            sal1, sal2, sal3, sal4, output = model(clip_s)
        
        annt_s=annt_s.to(dev)
        
        loss1 = criterion(sal1, annt_s)
        loss2 = criterion(sal2, annt_s)
        loss3 = criterion(sal3, annt_s)
        loss4 = criterion(sal4, annt_s)
        loss_out = criterion(output, annt_s)
        loss_sum = loss_sum + loss1.item() + loss2.item() + loss3.item() + loss4.item() + loss_out.item()
        loss1_sum += loss1.item()
        loss2_sum += loss2.item()
        loss3_sum += loss3.item()
        loss4_sum += loss4.item()
        lossOut_sum += loss_out.item()
        
        loss = loss1 + loss2 + loss3 + loss4 + loss_out
        loss.backward()

        
        if(i+1) % pile == 0:
           
            optimizer.step()
            
            optimizer.zero_grad()
            check_point['step']+=1
            
            print('Iteration: [%4d/%4d], Model Base: loss: %.4f, %s' % (check_point['step'], num_iters, loss_sum/pile, timedelta(seconds=int(time.time()-start_time))), flush=True)
            check_point['loss_history']['train'].append(loss_sum/pile)
            check_point['loss_history']['train_sal1'].append(loss1_sum/pile)
            check_point['loss_history']['train_sal2'].append(loss2_sum/pile)
            check_point['loss_history']['train_sal3'].append(loss3_sum/pile)
            check_point['loss_history']['train_sal4'].append(loss4_sum/pile)
            check_point['loss_history']['train_out'].append(lossOut_sum/pile)
            loss_sum = 0
            loss1_sum = 0 
            loss2_sum = 0
            loss3_sum = 0
            loss4_sum = 0 
            lossOut_sum = 0
            
            if check_point['step'] % num_val_iter == 0:
                model.eval()
                
                # select a random subset of indixes
                random_indexes = random.sample(range(len(valSet)),num_videos_val)
                
                loss_val_sum = 0
                for ri in tqdm(random_indexes, desc="***Validation***"):
                    
                    list_frames = valSet_list[ri]['list_frames']
                    list_annt = valSet_list[ri]['list_annts']
                    original_length= valSet_list[ri]['length']
                    saliency_map=[None]*original_length
            
                    #if number of video frames are less of 2*lentemporal, we append the frames to the list in reverse order
                    if original_length<2*len_temporal-1:
                        num_missed_frames =  2*len_temporal -1 - original_length
                        for k in range(num_missed_frames):
                            list_frames.append(np.copy(list_frames[original_length-k-1]))
                    
                    if len(list_frames) >=2*len_temporal-1:
                        snippet = []
                        
                        for i in range(len(list_frames)):
                            snippet.append(list_frames[i])
                            if i>= (len_temporal-1):
                                if i < original_length: #only for the original frames
                                    clip = transform(snippet)
                                    clip=clip.to(dev)
                                    with torch.set_grad_enabled(False):
                                        _,_,_,_,saliency_map[i]=model(clip)
                                        
                                if (i<2*len_temporal-2):
                                    j=i-len_temporal+1
                                    flipped_clip = torch.flip(clip, [1])
                                    with torch.set_grad_enabled(False):
                                        _,_,_,_,saliency_map[j]=model(flipped_clip)
                                        
                                del snippet[0]
                    
                    tens_saliency_map = torch.stack(saliency_map).to(dev)
                    tens_annt = torch.stack(list_annt).to(dev)
                    l = criterion(tens_saliency_map, tens_annt)
                    
                    loss_val_sum += l.item()
                    
                loss_val = loss_val_sum/num_videos_val
                check_point['loss_history']['validation'].append(loss_val)
                
                if loss_val < check_point['MIN_loss_val'] :
                    check_point['MIN_loss_val']  = loss_val
                    check_point['step_MIN_loss'] = check_point['step']
                    torch.save(model.state_dict(), os.path.join(path_output, 'weigth_MinLoss.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(path_output, 'adam_MinLoss.pt'))
                print('Model: step: %d, loss_val: %.4f, MIN_loss_val: %.4f, step_min_loss: %d' %(check_point['step'], loss_val, check_point['MIN_loss_val'] , check_point['step_MIN_loss']))
                
                model.train()
                '''
                    SAVE WEIGHTs AND CHECKPOINT
                '''
                torch.save(model.state_dict(), os.path.join(path_output, 'weight.pt'))
                torch.save(optimizer.state_dict(), os.path.join(path_output, 'adam.pt'))
                
                check_point['exec_time']=str(timedelta(seconds=int(time.time()-start_time)))
                with open(os.path.join('output', subfolder, test_name, 'check_point.json'), 'w') as fp:
                    json.dump(check_point, fp)
                    
                '''
                Plot Graph
                '''
                   
                # Plot loss
                x = torch.arange(1, len(check_point['loss_history']['train'])+1).numpy()
                plt.plot(x, check_point['loss_history']['train'], label="train_loss")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'loss.png'))
                plt.close()
                #plt.show()
                x = torch.arange(1, len(check_point['loss_history']['train'])+1).numpy()
                plt.plot(x, check_point['loss_history']['train_sal1'], label="train_loss salMap1")
                plt.plot(x, check_point['loss_history']['train_sal2'], label="train_loss salMap2")
                plt.plot(x, check_point['loss_history']['train_sal3'], label="train_loss salMap3")
                plt.plot(x, check_point['loss_history']['train_sal4'], label="train_loss salMap4")
                plt.plot(x, check_point['loss_history']['train_out'], label="train_loss Out")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'loss_singleSaliencyMaps.png'))
                plt.close()
                
                # Plot loss validation
                ax = torch.arange(1, len(check_point['loss_history']['validation'])+1).numpy()
                plt.plot(ax, check_point['loss_history']['validation'], label="validation_loss")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'loss_validation.png'))
                plt.close()
                
    info = ['train_time: ', str(timedelta(seconds=int(time.time()-start_time)))]
    file_info=open(os.path.join("output", subfolder, test_name, "train.txt"), 'a+', encoding='utf-8')
    file_info.writelines(info)
    file_info.close()
  
        
def transform(snippet):
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4) 
    return snippet   
                
if __name__ == '__main__':
    main()