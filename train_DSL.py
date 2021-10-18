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


from models.HD2S_DSL import HD2S_DSL
from loss import KLDLoss1vs1
from dataset.videoDataset import Dataset3D
from dataset.infiniteDataLoader import InfiniteDataLoader


source_datasets = [{'source': 'DHF1K', 'path': os.path.join('data','DHF1K','source')}, 
            {'source': 'Hollywood', 'path': os.path.join('data','Hollywood2','train')},
            {'source': 'UCFSports', 'path': os.path.join('data','UCF','train')}]

validation_datasets = [{'source': 'DHF1K', 'path': os.path.join('data','DHF1K','source')},
            #{'source': 'Hollywood', 'path': os.path.join('data','Hollywood2','train')},
            #{'source': 'UCFSports', 'path': os.path.join('data','UCF','train')}
            ]


def main():
    
    dev_name = 'cuda:0'
    
    pile = 25
    batch_size = 8
    len_temporal = 16
    validation_frac = 0.5
    
    image_size=(128,192)
    
    num_iters = 5000
    num_workers = 2
    lr=0.001
    
    num_val_iter = 100
    encoder_pretrained= True
    file_weight='none'
    
    sources=list(map(lambda x : x['source'], source_datasets))
    
    '''
    Model Parameters
    '''
    dict_model_params={
        'n_gaussian' : 0,
        'domSpec_bn' :False,
        'gaussian_layer' : False,
        'gaussian_priors' : False,
        'max_sigma' : 10,
        'activate_GL' : False
        }
    
    
    test_name= 'HD2S_DSL_training_demo_1'
    
    subfolder=os.path.join('DSL')
    
    path_source_data = list(map(lambda x : x['path'], source_datasets))
    source_loader=[None]*len(path_source_data)
    
    #validation on DHF1K
    list_path_validation = list(map(lambda x : x['path'], validation_datasets))
    list_source_validation =  list(map(lambda x : x['source'], validation_datasets))
    
    path_output = os.path.join('output','model_weights',subfolder, test_name)
    
    for idx, p in enumerate(path_source_data):
        
        print(p)
        
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
    weight_folder='HD2S_DSL_training_demo_1'
    weight_name='HD2S_DSL_weigths_MinLoss.pt'
    file_weight=os.path.join('output','model_weights',subfolder,weight_folder,weight_name)
    
    optim_name='adam.pt'
    file_optimizer=os.path.join('output','model_weights',subfolder,weight_folder,optim_name)
    '''
    
    model = HD2S_DSL(pretrained=encoder_pretrained,n_gaussians=dict_model_params['n_gaussian'], 
                                            sources=sources, domSpec_bn =dict_model_params['domSpec_bn'], gaussian_priors =dict_model_params['gaussian_priors'],
                                            gaussian_layer = dict_model_params['gaussian_layer'], max_sigma = dict_model_params['max_sigma'])
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
    
    #saving traing info
    summ = summary(model, input_data=(3, 16, 128, 192), device= dev, verbose=0)
    
    info=['model_name: ', model.__class__.__name__ ,'\n',
          'model_parameters: ', str(dict_model_params), '\n',
          'path_source_data_train: ', str(path_source_data), '\n',
          'path_validation: ', str(list_path_validation), '\n',
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
    
    dict_perDatasetLoss = {}
    sigma_dict={}
    dict_sum_perDatasetLoss ={}
    for s in sources:
        dict_perDatasetLoss[f'train_{s}']=[]
        dict_sum_perDatasetLoss[f'train_{s}_sum'] = 0
        dict_sum_perDatasetLoss[f'numBatch_train_{s}'] = 0
        sigma_dict[f'GL__{s.lower()}']=[]
    for d in list_source_validation:
        dict_perDatasetLoss[f'val_{d}']=[]
        dict_sum_perDatasetLoss[f'val_{d}_sum'] = 0
        
    check_point={ 'step' : 0,
                 'MIN_loss_val' : sys.float_info.max,
                 'step_MIN_loss' : 0,
                 'exec_time' : 0,
                 'sigma' : sigma_dict,
                 'loss_history' : {'train_sal1':[], 'train_sal2':[], 'train_sal3':[], 'train_sal4':[], 'train_out':[] ,'train':[], 'validation':[]},
                 'per_dataset_loss': dict_perDatasetLoss
        }
    
    '''
    with open(os.path.join('output', subfolder, test_name, 'check_point.json')) as fp:
        check_point=json.load(fp)
    '''
        
    '''
    preparation dict for validation set
    '''
    list_valSets=[]
    list_numVideosVal = []
    
    for path_validation in list_path_validation:
        path_val_split = os.path.join(path_validation, 'splitTrainVal','valSet2.csv')
        valSet = pd.read_csv(path_val_split, dtype = str)['0'].values.tolist()
        
        print('Preparation validation set...')
        valSet_list = []
        for v in tqdm(valSet):
            path_frames= os.path.join(path_validation, 'frames', v)
            if 'DHF1K' in path_validation:
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
        print('Done.')
        list_valSets.append(valSet_list)
        num_videos_val = int(len(valSet)*validation_frac)
        print(f'validation: {path_validation}, num_video_val:{num_videos_val}')
        list_numVideosVal.append(num_videos_val)
       
    #model losses
    loss_sum = 0
    loss1_sum = 0 
    loss2_sum = 0
    loss3_sum = 0
    loss4_sum = 0 
    lossOut_sum = 0
    
    index_source = -1
    
    
    for i in tqdm(range(check_point['step']*pile,num_iters*pile)):
        
        index_source = (index_source+1) % (len(source_datasets))
        
        clip_s, annt_s = next(source_loader[index_source])
        
        for name, param in model.named_parameters():
            for source in sources:
                if source.lower() in name.lower():
                    param.requires_grad = (source == sources[index_source])
            
        clip_s = clip_s.to(dev)
        sal1, sal2, sal3, sal4, output = model(clip_s, sources[index_source], dict_model_params['activate_GL'])
        
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
        
        dict_sum_perDatasetLoss[f'train_{sources[index_source]}_sum'] += loss_out.item()
        dict_sum_perDatasetLoss[f'numBatch_train_{sources[index_source]}'] += 1
        
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
            for source in sources:
                check_point['per_dataset_loss'][f'train_{source}'].append(dict_sum_perDatasetLoss[f'train_{source}_sum']/dict_sum_perDatasetLoss[f'numBatch_train_{source}'])
            for name,param in model.named_modules():
                if 'GL' in name:
                    check_point['sigma'][name].append(param.sigma.item())
            loss_sum = 0
            loss1_sum = 0 
            loss2_sum = 0
            loss3_sum = 0
            loss4_sum = 0 
            lossOut_sum = 0
            for source in sources:
                dict_sum_perDatasetLoss[f'train_{source}_sum'] = 0
                dict_sum_perDatasetLoss[f'numBatch_train_{source}'] = 0
            
            if check_point['step'] % num_val_iter == 0:
                
                
                '''******************BEGIN VALIDATION*********************'''
                print('*****************Validation********************')
                model.eval()
                loss_val_sum = 0
                for idx_val in range(len(list_path_validation)):
                    
                    # select a random subset of indixes
                    random_indexes = random.sample(range(len(list_valSets[idx_val])),list_numVideosVal[idx_val])

                    for ri in tqdm(random_indexes): #for each selected video calculate saliency map and loss
                        
                        list_frames = list_valSets[idx_val][ri]['list_frames']
                        list_annt = list_valSets[idx_val][ri]['list_annts']
                        original_length= list_valSets[idx_val][ri]['length']
                        saliency_map=[None]*original_length
                
                        #if number of video frames are less of 2*lentemporal, we append the frames to the list by going back
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
                                            _,_,_,_,saliency_map[i]=model(clip, list_source_validation[idx_val], dict_model_params['activate_GL'])
                                            
                                    if (i<2*len_temporal-2):
                                        j=i-len_temporal+1
                                        flipped_clip = torch.flip(clip, [1])
                                        with torch.set_grad_enabled(False):
                                            _,_,_,_,saliency_map[j]=model(flipped_clip,list_source_validation[idx_val], dict_model_params['activate_GL'])
                                            
                                    del snippet[0]
                        
                        tens_saliency_map = torch.stack(saliency_map).to(dev)
                        tens_annt = torch.stack(list_annt).to(dev)
                        l = criterion(tens_saliency_map, tens_annt)
                        
                        loss_val_sum += l.item()
                        dict_sum_perDatasetLoss[f'val_{list_source_validation[idx_val]}_sum'] += l.item()
                
                for idx,v in enumerate(list_source_validation):
                    check_point['per_dataset_loss'][f'val_{v}'].append(dict_sum_perDatasetLoss[f'val_{list_source_validation[idx]}_sum']/list_numVideosVal[idx])
                    dict_sum_perDatasetLoss[f'val_{v}_sum'] = 0
                
                loss_val = loss_val_sum/sum(list_numVideosVal)
                check_point['loss_history']['validation'].append(loss_val)
                
                if loss_val < check_point['MIN_loss_val'] :
                    check_point['MIN_loss_val']  = loss_val
                    check_point['step_MIN_loss'] = check_point['step']
                    torch.save(model.state_dict(), os.path.join(path_output, 'weight_MinLoss.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(path_output, 'adam_MinLoss.pt'))
                               
                print('Model: step: %d, loss_val: %.4f, MIN_loss_val: %.4f, step_min_loss: %.4f' %(check_point['step'], loss_val, check_point['MIN_loss_val'] , check_point['step_MIN_loss']))
                print('*********End Validation***********')
                

                '''******************END VALIDATION*********************'''
                
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
                
                #Plot train loss Per-Dataset
                x = torch.arange(1, len(check_point['per_dataset_loss'][f'train_{sources[0]}'])+1).numpy()
                for s in sources:
                    plt.plot(x, check_point['per_dataset_loss'][f'train_{s}'], label=f"train_loss {s}")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'trainLoss_perDataset.png'))
                plt.close()
                
                # Plot loss validation
                ax = torch.arange(1, len(check_point['loss_history']['validation'])+1).numpy()
                plt.plot(ax, check_point['loss_history']['validation'], label="validation_loss")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'loss_validation.png'))
                plt.close()
                
                #Plot validation per-dataset loss 
                x = torch.arange(1, len(check_point['per_dataset_loss'][f'val_{list_source_validation[0]}'])+1).numpy()
                for s in list_source_validation:
                    plt.plot(x, check_point['per_dataset_loss'][f'val_{s}'], label=f"val_loss {s}")
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'valLoss_perDataset.png'))
                plt.close()
                
                # Plot sigma
                ax = torch.arange(1, len(check_point['sigma'][f'GL__{list_source_validation[0].lower()}'])+1).numpy()
                for s in list_source_validation:
                    plt.plot(ax, check_point['sigma'][f'GL__{s.lower()}'], label=f'GL__{s.lower()}')
                plt.legend()
                plt.savefig(os.path.join('output', subfolder, test_name,'GL_sigma.png'))
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