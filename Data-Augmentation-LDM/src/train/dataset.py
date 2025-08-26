import os
import random
import cv2
import numpy as np
import torch
import scipy.io as sio

from torch.utils.data import Dataset

from .util import *

import sys
sys.path.append("...")

import glob
from audio.tools import wav_to_fbank, read_wav_file
from audio.stft import TacotronSTFT

from annotator.style.model import style_net
from annotator.style_pretrained.model import style_encoder


class UniDataset(Dataset):
    def __init__(self,
                 #anno_path,
                 #image_dir,
                 audio_dir,
                 #condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        #file_ids, self.annos = read_anno(anno_path)
        #self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]
        
        self.annos = []
        self.audio_paths = []
        count = 0

        '''
        mids = glob.glob(audio_dir+'/*')
        for mid in mids:
            vids = glob.glob(mid+'/SP_1/*')
            for vid in vids:
                self.annos.append("Damaged fuel pump sound")
                self.audio_paths.append(vid)
                count = count + 1
                print(count)
        '''
        
        vids = glob.glob(audio_dir+'/*')
        for vid in vids:
            if count < 0:
                count = count + 1
                continue
            if count >= 45000:
                break
            if len(glob.glob(vid+'/*')) == 0:
                continue
            cap = glob.glob(vid+'/*')[0]
            if len(glob.glob(cap+'/*')) == 0:
                continue
            if os.path.getsize(glob.glob(cap+'/*')[0]) < 1000000:
                continue
            if glob.glob(cap+'/*')[0][-4:] != '.wav':
                continue
            self.annos.append(cap[len(vid)+1:])
            self.audio_paths.append(glob.glob(cap+'/*')[0])
            count = count + 1
            print(count)
        

        #self.annos, self.audio_paths = [cap[len(audio_dir):], glob.glob(cap+'/*')[0] for cap in caps]
        
#         self.local_paths = {}
#         for local_type in local_type_list:
#             self.local_paths[local_type] = [os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids]
#         self.global_paths = {}
#         for global_type in global_type_list:
#             self.global_paths[global_type] = [os.path.join(condition_root, global_type, file_id + '.npy') for file_id in file_ids]
        
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob

        # condition extractors
        self.style = 'random'

        if self.style == 'random':
            self.audio_style_extractor = style_net()
            self.audio_style_extractor.load_state_dict(torch.load('annotator/style/style.ckpt'))
            #self.audio_style_extractor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        elif self.style == 'pretrained':
            self.audio_style_extractor = style_encoder()
        
        self.fn_STFT = TacotronSTFT(
            1024,
            160,
            1024,
            64,
            16000,
            0,
            8000,
        )
        
        self.duration = 10
    
    def __getitem__(self, index):
#         print('******path: ',self.audio_paths[index])
        mel, _, _ = wav_to_fbank(
            self.audio_paths[index], target_length=int(self.duration * 102.4), fn_STFT=self.fn_STFT
        )
        mel = mel.unsqueeze(0)

#         sio.savemat("mel.mat", {'mel':mel.cpu().detach().numpy()})
#         haha
        
#         torch.save(mel, './output/mel.pt')
#         haha
        
        '''
        image = cv2.imread(self.audio_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0
        '''
        
        waveform = read_wav_file(self.audio_paths[index], int(self.duration * 102.4) * 160)
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)
        
#         sio.savemat("waveform.mat", {'waveform':waveform.cpu().detach().numpy()})
#         haha

        if random.random() < self.drop_txt_prob:
            waveform = torch.zeros(waveform.shape)
        
        anno = self.annos[index]
        
        '''
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])
        global_files = []
        for global_type in self.global_type_list:
            global_files.append(self.global_paths[global_type][index])
        '''
        local_conditions = []
        '''
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
            local_conditions.append(condition)
        '''
        global_conditions = []
        '''
        for global_file in global_files:
            condition = np.load(global_file)
            global_conditions.append(condition)
        '''
        
        # audio style condition
        if self.style == 'random':
            style_condition = self.audio_style_extractor(mel.unsqueeze(0)).squeeze(0)
        elif self.style == 'pretrained':
            with torch.no_grad():
                style_condition = self.audio_style_extractor(self.audio_paths[index])

        # assert torch.isnan(style_condition.view(-1)).sum().item()==0
        
        # print('style_condition: ', style_condition.shape)

        global_conditions.append(style_condition.detach())
        
        if random.random() < self.drop_txt_prob:
            anno = ''
        
        local_conditions, local_condition_weights = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions, global_condition_weights = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)
            global_condition_weights = np.concatenate(global_condition_weights)

#         return dict(jpg=image, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions)
        return dict(fbank=mel, waveform=waveform, txt=anno, local_conditions=local_conditions, global_conditions=global_conditions, global_condition_weights=global_condition_weights)
        
    def __len__(self):
        return len(self.annos)
        
