import csv
import os
import os.path
import fnmatch

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import random
import re


import pandas as pd
import torchvision.transforms as transforms
import ast
import itertools
from collections import Counter
from scipy.sparse import coo_matrix

# emotion_categories = ['happiness','disgust','repression','surprise','fear','others','sadness']

class samm_dataset(data.Dataset):
    def __init__(self, data_path, label_path, sub_v, split, mask=None, transform=None):
        ############## use only onset, apex and offset ##################
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        
        if mask == None:
            # load data_file to df
            df = pd.read_csv(self.label_path,usecols=['Subject', 'Filename', 'Apex', 'Label'])
            
            # build emotion label dictionary 
            emotion_dic = {'Other': 0, 'Surprise': 1, 'Happiness': 2, 'Contempt': 3, 'Anger': 4}
            
            # build sequence of path 
            self.sequences = []                       
            
            for index,row in df.iterrows():
                
                sub = row['Subject']
                seq_name = row['Filename']
                label = row['Label'] 
                if label in ['Fear','Sadness','Disgust']:
                    continue;
                ### Three Frames Settings ###
                apex = row['Apex']
                if apex == '/':
                    continue;
                label = emotion_dic[label]
                seq_path = os.path.join(data_path,str(sub).zfill(3),seq_name)
                assert os.path.isdir(seq_path)
                

                if sub == sub_v and split =='test':
                    self.sequences.append((seq_path,label,apex))
                elif sub != sub_v and split =='train':
                    self.sequences.append((seq_path,label,apex))
                else:
                    continue;
           
        self.emotion_dic = emotion_dic
    
    def __getitem__(self, index):
        path, label, apex= self.sequences[index]
        for root, dirs, files in os.walk(path):
            for name in files:
                if str(apex)in name:
                    img_name = name
                else:
                    continue;
        img_name = os.path.join(path,img_name)
        assert os.path.isfile(img_name)
        #img_name = os.path.join(path,img_name)         
            
        normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                            std = [0.22803, 0.22145, 0.216989])  #standard of resnet 3d
        transform = transforms.Compose([transforms.Resize((112,112)),
                                            transforms.ToTensor(),
                                            normalize
                                            ])
        with Image.open(os.path.join(path,img_name)).convert("RGB") as img:
            if self.transform is None:
                img = transform(img)
            else:
                img = self.transform(img)
        label = torch.tensor(label,dtype = torch.long)

        '''# for one sequence, read all three images
        temp_name = []
        
        for frame in [on,apex,off]:
            for root, dirs, files in os.walk(path):
                for name in files:
                    
                    if str(frame)in name:
                        img_name = name
                    else:
                        continue;
            img_name = os.path.join(path,img_name)
            
            temp_name.append(img_name)
        

        seq_data = torch.zeros([len(temp_name),3,112,112])
        
        for index,img_name in enumerate(temp_name):
            normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                            std = [0.22803, 0.22145, 0.216989])  #standard of resnet 3d
            transform = transforms.Compose([transforms.Resize((112,112)),
                                            transforms.ToTensor(),
                                            normalize
                                            ])
            with Image.open(os.path.join(path,img_name)).convert("RGB") as img:
                if self.transform is None:
                    img = transform(img)
                else:
                    img = self.transform(img)
                
                seq_data[index]=img
        seq_data = seq_data.permute(1,0,2,3) ############################################
        label = torch.tensor(label,dtype = torch.long)
        '''
        

        return img, label, path# ,img_name
    
    
    
    def __len__(self):
        return len(self.sequences)





'''sub_v = random.randint(6, 27)
casme = samm_dataset(data_path='/home/lynn/Desktop/tmmocc/SAMM_masked',label_path="/home/lynn/Dataset/SAMM/SAMM.csv",sub_v=sub_v,split='test')'''