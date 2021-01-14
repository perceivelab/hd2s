import os
import numpy as np
import cv2
import torch
import pandas as pd

from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from PIL import Image


from models.SalGradNet import SalGradNet as modelName


dev = 'cuda:0'
image_size=(128, 192)

def main():
    target_dataset=os.path.join('DHF1K','validation')
    fromVideo=False
    
    len_temporal = 16
    
    test_name='SalGradNet_testDHF1K_demo'
    weight_folder='SalGradNet_train_demo'
    subfolder = 'BaseModel'
    weight_name='weight_MinLoss.pt'
    
    file_weight = os.path.join('output', 'model_weights', subfolder, weight_folder, weight_name)
    
    data_folder=os.path.join('data', target_dataset)
    video_folder=os.path.join('video')
    frames_folder='frames'
    path_output = os.path.join('output', subfolder, test_name)
    
    path_video=os.path.join(data_folder, video_folder)
    path_frames=os.path.join(data_folder, frames_folder)
    
    
    model=modelName()
    model.load_state_dict(torch.load(file_weight, map_location = dev))
    
    model=model.to(dev)
    torch.backends.cudnn.benchmark = True
    model.eval()

    if not os.path.isdir(os.path.join('output',subfolder,test_name)):
        os.makedirs(os.path.join('output',subfolder, test_name))

    if fromVideo:
        if target_dataset=='LEDOV' or target_dataset=='UAV123':
            list_video= pd.read_csv(os.path.join('data',target_dataset,'test.csv'))['0'].values.tolist()
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
            
            # if number of video frames are less of 2*lentemporal, we append the frames to the list in reverse order
            if original_length<2*len_temporal-1:
                num_missed_frames =  2*len_temporal -1 - original_length
                for k in range(num_missed_frames):
                    list_frames.append(np.copy(list_frames[original_length-k-1]))
            
            # process in a sliding window fashion
            if len(list_frames) >= 2*len_temporal-1:
        
                frames_mask=[None]*original_length
                overlap=[None]*original_length
    
                snippet = []
                for i in tqdm(range(len(list_frames)), desc=f"number of frames: {len(list_frames)}"):
                    img = list_frames[i]
                   
                    snippet.append(img)
                    if i<original_length:
                        overlap[i]=Image.fromarray(np.uint8(list_frames[i]), "RGB")
                    
                    if (i>= len_temporal -1):
                        
                        if i < original_length:#only for the original frames
                            clip = transform(snippet)
                            frames_mask[i]=process(model, clip, i, destination_path)
                        
                            img = cv2.applyColorMap(frames_mask[i],cv2.COLORMAP_HOT)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        
                            overlap[i].paste(Image.fromarray(img), mask=Image.fromarray(frames_mask[i]))
                            
                        if (i<2*len_temporal-2):
                            j=i-len_temporal+1
                            frames_mask[j] = process(model, torch.flip(clip, [1]), j, destination_path)
                            img = cv2.applyColorMap(frames_mask[j],cv2.COLORMAP_HOT)
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            overlap[j].paste(Image.fromarray(img), mask=Image.fromarray(frames_mask[j]))
                        del snippet[0]
                        
                if not os.path.isdir(os.path.join(destination_path,'images')):
                        os.mkdir(os.path.join(destination_path,'images'))
                for idx in range(len(overlap)):
                    overlap[idx].save(os.path.join(destination_path, 'images', '%04d.jpg'%(idx+1)), format='JPEG', quality=100)
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

     

def process(model, clip, idx, path_output):
    frames_path = os.path.join(path_output,'frames')
    if not os.path.isdir(frames_path):
        os.mkdir(frames_path)
        
    with torch.no_grad():
        _,_,_,_, smap = model(clip.to(dev))
        
    smap=smap.cpu().data[0]
    
    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=5)
    smap = (smap/np.max(smap)*255.).astype(np.uint8)
    cv2.imwrite(os.path.join(frames_path, '%04d.png'%(idx+1)), smap)
    
    return smap

if __name__ == '__main__':
    main()

