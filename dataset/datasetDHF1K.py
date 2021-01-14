import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

def transform3D(snippet):
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
    return snippet

class Dataset3D(Dataset):
    def __init__(self, path_data, len_snippet, size=(224,384), target = False, list_videoName=[]):
         self.path_data = path_data 
         self.len_snippet = len_snippet 
         self.size=(size[1], size[0])
         self.target = target
         self.list_video_frames=[] #it contains one element for each video; each element is a list of frame names of that video
         self.list_video_annt=[] #it contains one element for each video; each element is a list of annt names of that video
         self.list_num_frame=[] #it contains one element for each video; each element is the number of frames of that video
         
         print("Init Dataset...")
         if len(list_videoName)>0:
             self.list_video_name=list_videoName
             self.list_video_name=[v for v in self.list_video_name]
             self.list_video_name.sort()
         elif path_data==os.path.join('data', 'LEDOV'):
             self.list_video_name=pd.read_csv(os.path.join(path_data,'train.csv'))['0'].values.tolist()
             self.list_video_name=[os.path.splitext(v)[0] for v in self.list_video_name]
             self.list_video_name.sort()
         else:
             self.list_video_name=[d for d in os.listdir(os.path.join(path_data, 'frames')) if os.path.isdir(os.path.join(path_data, 'frames', d))]
        
        
         for vid in self.list_video_name:
             #list of frame names of a single video
             list_frame_names=[f for f in os.listdir(os.path.join(path_data, 'frames', vid)) if os.path.isfile(os.path.join(path_data, 'frames', vid, f))]
             list_frame_names.sort()
             if self.target is False:
                 #list of annt names of a single video 
                 if self.path_data==os.path.join('data', 'DHF1K', 'source'):
                     vid='%04d'%(int(vid))
                 list_annt_names=[a for a in os.listdir(os.path.join(path_data, 'annotation', vid, 'maps')) if os.path.isfile(os.path.join(path_data, 'annotation', vid, 'maps',a))]
                 list_annt_names.sort()
                 assert len(list_frame_names)==len(list_annt_names), 'something wrong with len of frames and annotation in datasetLoader'
                 self.list_video_annt.append(list_annt_names)
                 
             self.list_video_frames.append(list_frame_names)
             self.list_num_frame.append(len(list_frame_names))
         print("Init Dataset Completed.")
        

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        video_name = self.list_video_name[idx] 
        video_frames_path = os.path.join(self.path_data, 'frames', video_name)
        if self.target is False:
            if self.path_data==os.path.join('data', 'DHF1K', 'source'):
                video_name='%04d'%(int(video_name))
            video_annt_path = os.path.join(self.path_data, 'annotation', video_name, 'maps')
            list_annt_names=self.list_video_annt[idx]
            
        list_frame_names= self.list_video_frames[idx]
        
        
        #random index of frame in range [0,num_frame] of the selected video
        start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)
        v = np.random.random()
        clip = []
        last_index=0 #index of the last frame added in snippet
        for i in range(self.len_snippet):
            frame_path=os.path.join(video_frames_path, list_frame_names[start_idx+i])
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
            img = cv2.resize(img, self.size)
            if v < 0.5:
                img = img[:, ::-1, ...]
            clip.append(img)
            last_index=start_idx+i
        
        if self.target is False:
            annt_path=os.path.join(video_annt_path, list_annt_names[last_index])
            annt = cv2.imread(annt_path, 0)
            annt = cv2.resize(annt, self.size)
            if v < 0.5:
                annt = annt[:, ::-1]
        else:
            annt = np.zeros(self.size)
        return transform3D(clip), torch.from_numpy(annt.copy()).contiguous().float()
