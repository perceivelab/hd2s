import os
import torch
import time
import numpy as np
import pandas as pd
import sys
import cv2

from datetime import timedelta
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchsummary import summary

from models.HD2S_DA import HD2S_DA
from loss import KLDLoss1vs1
from dataset.videoDataset import Dataset3D
from dataset.infiniteDataLoader import InfiniteDataLoader



def main():
    
    dev = "cuda:0"
    
    pile = 25
    batch_size = 8
    len_temporal = 16
    image_size=(128,192)
    
    num_iters = 2500
    num_workers = 2
    lr=0.001
    
    num_val_iter = 100
    
    test_name= 'HD2S_DA_training_demo_1'
    subfolder = 'DA'
    #path_source_data = os.path.join('data','DHF1K','source')
    #path_target_data = os.path.join('data','UCF', 'train')
    path_source_data = os.path.join("C:\\","Users","gbellitto","Desktop","GitRepository","video-saliency-detection","data",'DHF1K','source')
    path_target_data= os.path.join("C:\\","Users","gbellitto","Desktop","GitRepository","video-saliency-detection","data",'UCF','train')
    
    path_train_split = os.path.join(path_source_data, 'splitTrainVal','trainSet2.csv')
    path_val_split = os.path.join(path_source_data, 'splitTrainVal','valSet2.csv')
    
    path_output = os.path.join('output','model_weights',subfolder,test_name)
    
    trainSet = pd.read_csv(path_train_split, dtype = str)['0'].values.tolist()
    valSet = pd.read_csv(path_val_split, dtype = str)['0'].values.tolist()
    
    if not os.path.isdir(os.path.join('output',test_name)):
        os.makedirs(os.path.join('output',test_name))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)    
    
    '''
    #loading weights file (fine-tuning)
    weight_folder='HDS_DA_training_demo_1'
    weight_name='weights_MinLoss.pt'
    file_weight=os.path.join('output','model_weights',subfolder,weight_folder,weight_name)
    
    optim_name='adam_MinLoss.pt'
    file_optimizer=os.path.join('output','model_weights',weight_folder,optim_name)
    '''
    
    model = HD2S_DA().to(dev)
    '''
    # loading file weight (fine-tuning)
    model.load_state_dict(torch.load(file_weight, map_location=dev))
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-7)
    
    '''
    #loading optimizer (fine-tuning)
    optim_name='adam_MinLoss.pt'
    '''   
    
    torch.backends.cudnn.benchmark = True
    
    criterion = KLDLoss1vs1()
    criterion_domain= torch.nn.NLLLoss()
    
    model.train()
    
    #saving traing info
    summ = summary(model, input_data=(3, 16, 128, 192), device= dev, verbose=0)
    
    info=['model_name: ', model.__class__.__name__ ,'\n',
          'path_source_data_train: ', path_source_data, '\n',
          'path_target_data_train: ', path_target_data,'\n',
          'pile ', str(pile),'\n',
          'batch_size: ', str(batch_size),'\n',
          'len_temporal: ', str(len_temporal),'\n',
          'image_size ',  str(image_size),'\n',
          'num_iters ',  str(num_iters),'\n',
          'num_workers ',  str(num_workers),'\n',
          'lr: ', str(lr), '\n',
          'num_val_iter', str(num_val_iter), '\n',
          'model_summary: ','\n', str(summ), '\n']
    
    file_info=open(os.path.join("output",subfolder,test_name, "train.txt"), 'w', encoding='utf-8')
    file_info.writelines(info)
    file_info.close()
    
    source_loader = InfiniteDataLoader(Dataset3D(path_source_data,len_temporal, size=image_size, list_videoName = trainSet), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    target_loader = InfiniteDataLoader(Dataset3D(path_target_data,len_temporal, size=image_size, target=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    start_time = time.time()
    
    step = 0
    
    #model with GRL and MultiCLassifier
    loss_s_saliency_sum=0
    loss_s_domain_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
    loss_t_domain_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
    accuracy_s_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
    accuracy_t_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
    loss_history={'saliency': [], 
                      's_domain':{'class5':[], 'class4':[], 'class3':[], 'class2':[]}, 
                      't_domain':{'class5':[], 'class4':[], 'class3':[], 'class2':[]},
                      'validation':[]}
    accuracy_history={'s_domain':{'class5':[], 'class4':[], 'class3':[], 'class2':[]}, 
                          't_domain':{'class5':[], 'class4':[], 'class3':[], 'class2':[]}}
    MIN_loss_val = sys.float_info.max
    step_MIN = 0
    
    for i in tqdm(range(step*pile,num_iters*pile)):
        
        #alpha parameter for GRL
        p = float( i/ (num_iters*pile))
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        clip_s, annt_s=next(source_loader)
        #source domain (domain 0)
        domain_label=(torch.zeros(batch_size)).long()
        domain_label=domain_label.to(dev)
        
        with torch.set_grad_enabled(True):
            clip_s = clip_s.to(dev)
            output_GRL,domain_out5, domain_out4, domain_out3, domain_out2 = model(clip_s, alpha)
            
        annt_s=annt_s.to(dev)
        #loss model withGRL and multiclassifier
        loss_s_saliency = criterion(output_GRL, annt_s)
        loss_s_domain_5=criterion_domain(domain_out5, domain_label)
        loss_s_domain_4=criterion_domain(domain_out4, domain_label)
        loss_s_domain_3=criterion_domain(domain_out3, domain_label)
        loss_s_domain_2=criterion_domain(domain_out2, domain_label)
        
        loss_s_saliency_sum += loss_s_saliency.item()
        loss_s_domain_sum['class5'] += loss_s_domain_5.item()
        loss_s_domain_sum['class4'] += loss_s_domain_4.item()
        loss_s_domain_sum['class3'] += loss_s_domain_3.item()
        loss_s_domain_sum['class2'] += loss_s_domain_2.item()
        
        #accuracy model with GRL and MultiClassifier
        _, pred_s5 = domain_out5.max(1)
        correct_s_GRL5 = pred_s5.eq(domain_label).sum().item()
        accuracy_s_sum['class5'] += correct_s_GRL5 / batch_size
        
        _, pred_s4 = domain_out4.max(1)
        correct_s_GRL4 = pred_s4.eq(domain_label).sum().item()
        accuracy_s_sum['class4'] += correct_s_GRL4 / batch_size
        
        _, pred_s3 = domain_out3.max(1)
        correct_s_GRL3 = pred_s3.eq(domain_label).sum().item()
        accuracy_s_sum['class3'] += correct_s_GRL3 / batch_size
        
        _, pred_s2 = domain_out2.max(1)
        correct_s_GRL2 = pred_s2.eq(domain_label).sum().item()
        accuracy_s_sum['class2'] += correct_s_GRL2 / batch_size
        
        
        #target_domain (domain 1)
        domain_label=(torch.ones(batch_size)).long()
        domain_label=domain_label.to(dev)
        
        clip_t, annt_t=next(target_loader)
        
        with torch.set_grad_enabled(True):
            _,domain_out5, domain_out4, domain_out3, domain_out2 = model(clip_t.to(dev), alpha)
        
        #loss model with GRL and MultiClassifier
        loss_t_domain_5=criterion_domain(domain_out5, domain_label)
        loss_t_domain_4=criterion_domain(domain_out4, domain_label)
        loss_t_domain_3=criterion_domain(domain_out3, domain_label)
        loss_t_domain_2=criterion_domain(domain_out2, domain_label)
        
        loss_t_domain_sum['class5'] += loss_t_domain_5.item()
        loss_t_domain_sum['class4'] += loss_t_domain_4.item()
        loss_t_domain_sum['class3'] += loss_t_domain_3.item()
        loss_t_domain_sum['class2'] += loss_t_domain_2.item()
        
        #accuracy model woth GRL and MultiClassifier
        _, pred_t5 = domain_out5.max(1)
        correct_t_GRL5 = pred_t5.eq(domain_label).sum().item()
        accuracy_t_sum['class5'] += correct_t_GRL5 / batch_size
        
        _, pred_t4 = domain_out4.max(1)
        correct_t_GRL4 = pred_t4.eq(domain_label).sum().item()
        accuracy_t_sum['class4'] += correct_t_GRL4 / batch_size
        
        _, pred_t3 = domain_out3.max(1)
        correct_t_GRL3 = pred_t3.eq(domain_label).sum().item()
        accuracy_t_sum['class3'] += correct_t_GRL3 / batch_size
        
        _, pred_t2 = domain_out2.max(1)
        correct_t_GRL2 = pred_t2.eq(domain_label).sum().item()
        accuracy_t_sum['class2'] += correct_t_GRL2 / batch_size
        
        
        
        loss_s_saliency = loss_s_saliency.to(dev)
        error = loss_s_saliency + loss_s_domain_5+ loss_s_domain_4 + loss_s_domain_3 + loss_s_domain_2 + loss_t_domain_5 + loss_t_domain_4 + loss_t_domain_3 + loss_t_domain_2
        
        error.backward()
    
        if(i+1) % pile == 0:
            optimizer.step()
            optimizer.zero_grad()
            step+=1
            
            #model with GRL and MultiClassifier
            print('Iteration: [%4d/%4d], HD2S_DA: alpha:%.4f,  %s' % (step, num_iters, alpha, timedelta(seconds=int(time.time()-start_time))), flush=True)
            loss_history['saliency'].append(loss_s_saliency_sum/pile)
            loss_history['s_domain']['class5'].append(loss_s_domain_sum['class5']/pile)
            loss_history['s_domain']['class4'].append(loss_s_domain_sum['class4']/pile)
            loss_history['s_domain']['class3'].append(loss_s_domain_sum['class3']/pile)
            loss_history['s_domain']['class2'].append(loss_s_domain_sum['class2']/pile)
            loss_history['t_domain']['class5'].append(loss_t_domain_sum['class5']/pile)
            loss_history['t_domain']['class4'].append(loss_t_domain_sum['class4']/pile)
            loss_history['t_domain']['class3'].append(loss_t_domain_sum['class3']/pile)
            loss_history['t_domain']['class2'].append(loss_t_domain_sum['class2']/pile)
            accuracy_history['s_domain']['class5'].append(accuracy_s_sum['class5']/pile)
            accuracy_history['s_domain']['class4'].append(accuracy_s_sum['class4']/pile)
            accuracy_history['s_domain']['class3'].append(accuracy_s_sum['class3']/pile)
            accuracy_history['s_domain']['class2'].append(accuracy_s_sum['class2']/pile)
            accuracy_history['t_domain']['class5'].append(accuracy_t_sum['class5']/pile)
            accuracy_history['t_domain']['class4'].append(accuracy_t_sum['class4']/pile)
            accuracy_history['t_domain']['class3'].append(accuracy_t_sum['class3']/pile)
            accuracy_history['t_domain']['class2'].append(accuracy_t_sum['class2']/pile)
            loss_s_saliency_sum=0
            loss_s_domain_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
            loss_t_domain_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
            accuracy_s_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
            accuracy_t_sum={'class5': 0.0, 'class4': 0.0, 'class3': 0.0, 'class2':0.0}
            
            
            if step % num_val_iter == 0:
                torch.save(model.state_dict(), os.path.join(path_output, 'weigth.pt'))
                torch.save(optimizer.state_dict(), os.path.join(path_output, 'adam.pt'))
                file_step=open(os.path.join('output', test_name, 'step.txt'), 'w', encoding='utf-8')
                file_step.writelines(str(step))
                file_step.close()
                
                '''******************BEGIN VALIDATION*********************'''
                print('*****************Validation********************')
                model.eval()
                
                loss_val_sum = 0
                for v in tqdm(valSet):
                    path_frames= os.path.join(path_source_data, 'frames', v)
                    if 'DHF1K' in path_source_data:
                        path_annt = os.path.join(path_source_data, 'annotation', "%04d"% int(v), 'maps')
                    else:    
                        path_annt = os.path.join(path_source_data, 'annotation', v, 'maps')
                    list_frame_names = [f for f in os.listdir(path_frames) if os.path.isfile(os.path.join(path_frames, f))]
                    list_annt_names = [a for a in os.listdir(path_annt) if os.path.isfile(os.path.join(path_annt, a))]
                    list_frames = []
                    list_annt = []
                    original_length= len(list_frame_names)
                    saliency_map=[None]*original_length
                    
                    for f in list_frame_names:
                        img = cv2.imread(os.path.join(path_frames, f))
                        img= cv2.resize(img, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
                        list_frames.append(img)
                        
                    for a in list_annt_names:
                        img = cv2.imread(os.path.join(path_annt, a), 0)
                        img= cv2.resize(img, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
                        list_annt.append(torch.from_numpy(img.copy()).contiguous().float())
            
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
                                        saliency_map[i] = model(clip)
                                
                                if (i<2*len_temporal-2):
                                    j=i-len_temporal+1
                                    flipped_clip = torch.flip(clip, [1])
                                    with torch.set_grad_enabled(False):
                                        saliency_map[j] = model(flipped_clip)
                                
                                del snippet[0]

                    tens_saliency_map = torch.stack(saliency_map).to(dev)
                    tens_annt = torch.stack(list_annt).to(dev)
                    ll = criterion(tens_saliency_map, tens_annt)
                    
                    loss_val_sum += ll.item()
                    
                loss_val = loss_val_sum/len(valSet)
                loss_history['validation'].append(loss_val)
                
                if loss_val < MIN_loss_val:
                    MIN_loss_val = loss_val 
                    step_MIN = step
                    min_info = ['min loss step: ', str(step_MIN), '\n', 'min loss value: ', str(MIN_loss_val)]
                    file_stepMin=open(os.path.join('output', test_name, 'stepMinLoss.txt'), 'w', encoding='utf-8')
                    file_stepMin.writelines(min_info)
                    file_stepMin.close()
                    torch.save(model.state_dict(), os.path.join(path_output, 'weigth_MinLoss.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(path_output, 'adam_MinLoss.pt'))
                
                print('HD2S_DA: step: %d, loss_val: %.4f, MIN_loss_val: %.4f, step_min_loss: %.4f' %(step, loss_val, MIN_loss_val, step_MIN))
                print('*********End Validation***********')
                
                model.train()
                '''******************END VALIDATION*********************'''
                
                # Plot loss
                x = torch.arange(1, len(loss_history['saliency'])+1).numpy()
                plt.plot(x, loss_history['saliency'], label="saliency_loss")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_saliency.png'))
                plt.close()
                #plt.show()
                
                #Loss Classifier
                plt.plot(x, loss_history['s_domain']['class5'], label="s_domain_loss5")
                plt.plot(x, loss_history['t_domain']['class5'], label="t_domain_loss5")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_domain5.png'))
                plt.close()
                
                plt.plot(x, loss_history['s_domain']['class4'], label="s_domain_loss4")
                plt.plot(x, loss_history['t_domain']['class4'], label="t_domain_loss4")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_domain4.png'))
                plt.close()
                
                plt.plot(x, loss_history['s_domain']['class3'], label="s_domain_loss3")
                plt.plot(x, loss_history['t_domain']['class3'], label="t_domain_loss3")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_domain3.png'))
                plt.close()
                
                plt.plot(x, loss_history['s_domain']['class2'], label="s_domain_loss2")
                plt.plot(x, loss_history['t_domain']['class2'], label="t_domain_loss2")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_domain2.png'))
                plt.close()
                
                #Loss Classifier All in One fig
                plt.plot(x, loss_history['s_domain']['class5'], label="s_domain_loss5")
                plt.plot(x, loss_history['t_domain']['class5'], label="t_domain_loss5")
                plt.plot(x, loss_history['s_domain']['class4'], label="s_domain_loss4")
                plt.plot(x, loss_history['t_domain']['class4'], label="t_domain_loss4")
                plt.plot(x, loss_history['s_domain']['class3'], label="s_domain_loss3")
                plt.plot(x, loss_history['t_domain']['class3'], label="t_domain_loss3")
                plt.plot(x, loss_history['s_domain']['class2'], label="s_domain_loss2")
                plt.plot(x, loss_history['t_domain']['class2'], label="t_domain_loss2")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_allClassifier.png'))
                plt.close()
                
                #Accuracy classifier
                plt.plot(x, accuracy_history['s_domain']['class5'], label="s_domain_acc5")
                plt.plot(x, accuracy_history['t_domain']['class5'], label="t_domain_acc5")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'accuracy_domain5.png'))
                plt.close()
                
                plt.plot(x, accuracy_history['s_domain']['class4'], label="s_domain_acc4")
                plt.plot(x, accuracy_history['t_domain']['class4'], label="t_domain_acc4")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'accuracy_domain4.png'))
                plt.close()
                
                plt.plot(x, accuracy_history['s_domain']['class3'], label="s_domain_acc3")
                plt.plot(x, accuracy_history['t_domain']['class3'], label="t_domain_acc3")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'accuracy_domain3.png'))
                plt.close()
                
                plt.plot(x, accuracy_history['s_domain']['class2'], label="s_domain_acc2")
                plt.plot(x, accuracy_history['t_domain']['class2'], label="t_domain_acc2")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'accuracy_domain2.png'))
                plt.close()
                
                #Accuracy Classifier All in One fig
                plt.plot(x, accuracy_history['s_domain']['class5'], label="s_domain_acc5")
                plt.plot(x, accuracy_history['t_domain']['class5'], label="t_domain_acc5")
                plt.plot(x, accuracy_history['s_domain']['class4'], label="s_domain_acc4")
                plt.plot(x, accuracy_history['t_domain']['class4'], label="t_domain_acc4")
                plt.plot(x, accuracy_history['s_domain']['class3'], label="s_domain_acc3")
                plt.plot(x, accuracy_history['t_domain']['class3'], label="t_domain_acc3")
                plt.plot(x, accuracy_history['s_domain']['class2'], label="s_domain_acc2")
                plt.plot(x, accuracy_history['t_domain']['class2'], label="t_domain_acc2")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'accuracy_allClassifier.png'))
                plt.close()
                
                # Plot loss validation
                ax = torch.arange(1, len(loss_history['validation'])+1).numpy()
                plt.plot(ax, loss_history['validation'], label="validation_loss")
                plt.legend()
                plt.savefig(os.path.join('output',test_name,'loss_validation.png'))
                plt.close()
        
def transform(snippet):
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float() 
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4) 
    return snippet   
                
if __name__ == '__main__':
    main()