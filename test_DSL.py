import os
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image

from models.HD2S_DSL import HD2S_DSL as modelName

source_datasets = [{'source': 'DHF1K', 'path': os.path.join('data','DHF1K','validation')}, 
            {'source': 'Hollywood', 'path': os.path.join('data','Hollywood2','test')},
            {'source': 'UCFSports', 'path': os.path.join('data','UCF','test')}]

dataset_index = 0
fromVideo=False
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size=(128, 192)

dataset_source = source_datasets[dataset_index]['source']
encoder_pretrained = False

'''
Model Parameters
'''
dict_model_params={
    'n_gaussian' : 16,
    'domSpec_bn' : True,
    'gaussian_layer' : True,
    'gaussian_priors' : True,
    'max_sigma' : 10
    }

def main():
    
    test_name=f'HD2S_testDHF1K_{dataset_source}_demo'
    weight_folder='HDS2_DSL_training_demo'
    weight_name='HD2S_DSL_weigths_MinLoss.pt'
    subfolder = 'DSL'
    len_temporal = 16
    
    file_weight = os.path.join('output', 'model_weights', subfolder, weight_folder, weight_name)
    
    data_folder = source_datasets[dataset_index]['path']
    video_folder=os.path.join('video')
    frames_folder='frames'
    path_output = os.path.join('output', subfolder, test_name)
    
    path_video=os.path.join(data_folder, video_folder)
    path_frames=os.path.join(data_folder, frames_folder)
    
    
    model=modelName(pretrained=encoder_pretrained,n_gaussians=dict_model_params['n_gaussian'], 
                    sources= [dataset_source], domSpec_bn =dict_model_params['domSpec_bn'], gaussian_priors =dict_model_params['gaussian_priors'],
                    gaussian_layer = dict_model_params['gaussian_layer'])
    model=model.to(dev)
    weight_dict = torch.load(file_weight, map_location = dev)
    model.load_state_dict(weight_dict, strict = False)
    
    torch.backends.cudnn.benchmark = True
    model.eval()
    
    if not os.path.isdir(os.path.join('output',subfolder,test_name)):
        os.makedirs(os.path.join('output',subfolder, test_name))
    
    #saving test info
    info=['model_name: ', model.__class__.__name__ ,'\n',
          'model_parameters: ', str(dict_model_params), '\n',
          'len_temporal: ', str(len_temporal),'\n',
          'image_size: ',  str(image_size),'\n',
          'file_weight: ', str(file_weight),'\n'
          ]
    file_info=open(os.path.join("output", subfolder, test_name, "info.txt"), 'w', encoding='utf-8')
    file_info.writelines(info)
    file_info.close()
    
    if fromVideo:
        if dataset_source=='LEDOV' or dataset_source=='UAV123':
            list_video= pd.read_csv(os.path.join('data',dataset_source,'test.csv'))['0'].values.tolist()
            list_video.sort()
        else:   
            list_video = [v for v in os.listdir(path_video) if os.path.isfile(os.path.join(path_video, v))] 
    else:
        list_video = [v for v in os.listdir(path_frames) if os.path.isdir(os.path.join(path_frames, v))] 
        
    for v in list_video:
        destination_path=os.path.join(path_output,os.path.splitext(v)[0])
        print(destination_path)
        if not os.path.isdir(destination_path):
            os.mkdir(destination_path)
            
            if fromVideo:
                list_frames = resized_frames_from_video(v, path_video)
            else:
                list_frame_names = [f for f in os.listdir(os.path.join(path_frames,v)) if os.path.isfile(os.path.join(path_frames, v,f))] 
                list_frames=[]
                for f in list_frame_names:
                    img = cv2.imread(os.path.join(path_frames,v, f))
                    img= cv2.resize(img, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
                    list_frames.append(img)
            
            original_length= len(list_frames)
            
            #if the number of video frames are less of 2*lentemporal, we append the frames to the list by going back
            if original_length<2*len_temporal-1:
                num_missed_frames =  2*len_temporal -1 - original_length
                for k in range(num_missed_frames):
                    list_frames.append(np.copy(list_frames[original_length-k-1]))
            
            # process in a sliding window fashion
            if len(list_frames) >= 2*len_temporal-1:
        
                frames_mask=[None]*original_length
                overlap=[None]*original_length
    
                snippet = []
                print(f"numbers of frames: {len(list_frames)}")
                for i in tqdm(range(len(list_frames))):
                    img = list_frames[i]
                   
                    snippet.append(img)
                    
                    if i<original_length:
                        overlap[i]=Image.fromarray(np.uint8(list_frames[i]), "RGB")
                    
                    if (i>= len_temporal -1):
                        
                        if i < original_length:#only for the original frames
                            clip = transform(snippet)
                            frames_mask[i]=process(model, dataset_source, clip, i, destination_path)
                            
                            img = cv2.applyColorMap(frames_mask[i],cv2.COLORMAP_HOT)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        
                            overlap[i].paste(Image.fromarray(img), mask=Image.fromarray(frames_mask[i]))
                            
                        if (i<2*len_temporal-2):
                            j=i-len_temporal+1
                            frames_mask[j] = process(model, dataset_source, torch.flip(clip, [1]), j, destination_path)
                            
                            img = cv2.applyColorMap(frames_mask[j],cv2.COLORMAP_HOT)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            overlap[j].paste(Image.fromarray(img), mask=Image.fromarray(frames_mask[j]))
                            
                        del snippet[0]
                        
                if not os.path.isdir(os.path.join(destination_path,'images')):
                        os.mkdir(os.path.join(destination_path,'images'))
                for idx in range(len(overlap)):
                    overlap[idx].save(os.path.join(destination_path, 'images', '%04d.jpg'%(idx+1)), format='JPEG', quality=100)
                
                '''
                #saving the gif file...
                v_name=os.path.splitext(v)[0]+'.gif'
                
                
                overlap[0].save(os.path.join(destination_path,v_name) , format='GIF',
                       append_images=overlap[1:],
                       save_all=True,
                       duration=40, loop=0)
                '''
            else: print("more frames are needed")



def resized_frames_from_video(v, path_video):
    print(os.path.join(path_video,v))
    vidcap = cv2.VideoCapture(os.path.join(path_video,v))
    success,image = vidcap.read()
    frames=[]
    success = True
    while success:
        image = cv2.resize(image, dsize=(image_size[1], image_size[0]),interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        frames.append(image)
        success,image = vidcap.read()
    return frames


def transform(snippet):
    snippet = np.concatenate(snippet, axis=-1)  
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()  
    snippet = snippet.mul_(2.).sub_(255).div(255)     
    snippet = snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4) 
    return snippet

     

def process(model, dataset_source, clip, idx, path_output):
    frames_path = os.path.join(path_output,'frames')
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)
        
    with torch.no_grad():
        _,_,_,_, smap = model(clip.to(dev), dataset_source)
    
    smap=smap.cpu().data[0].numpy()
    smap=(smap/np.max(smap)*255.).astype(np.uint8)
    cv2.imwrite(os.path.join(frames_path, '%04d.png'%(idx+1)), smap)
    
    return smap

if __name__ == '__main__':
    main()

