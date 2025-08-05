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

class samm_triplet_dataset(data.Dataset):
    def __init__(self, data_path, label_path, sub_v, split, transform=None):
        ############## use only onset, apex and offset ##################
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.split = split
        
        # load data_file to df
        df = pd.read_csv(self.label_path,usecols=['Subject', 'Filename', 'Apex', 'Label', 'Onset'])
        
        # build emotion label dictionary 
        emotion_dic = {'Other': 0, 'Surprise': 1, 'Happiness': 2, 'Contempt': 3, 'Anger': 4}
        drop_index  = df[(df['Label']=='Fear')|(df['Label']=='Sadness')|(df['Label']=='Disgust')  ].index
        df = df.drop(drop_index)
        self.emotion_dic = emotion_dic
        sub_list = set(df['Subject'].unique())
        
        # compute label weight
        value = torch.tensor(df.groupby("Label").count().sort_index().iloc[:, 0].values)
        self.weight = torch.div(torch.full_like(value, torch.max(value)), value)
        
        # build sequence of path 
        self.sequences = []                       
        
        for index,row in df.iterrows():
            
            sub = row['Subject']
            seq_name = row['Filename']
            label = row['Label'] 
            ### Apex Frame only Settings ###
            apex = row['Apex']
            if apex == '/':
                continue;
            ### Neutral Frames ###
            subA_neutral = row['Onset']
            label = emotion_dic[label]
            '''seq_path = os.path.join(data_path,str(sub).zfill(3),seq_name)
            assert os.path.isdir(seq_path)'''

            ### Random Another Subject ###          
            rand_sub = sub
            while str(rand_sub) == str(sub): ### choose another subject ###
                rand_sub = random.sample(sub_list,1)
                rand_sub = rand_sub[0]
            
            
            subB = df.loc[df['Subject'] == rand_sub].sample()
            
            subB_seq = subB['Filename'].values[0]
            subB_neutral = subB['Onset'].values[0]
            subB_path = os.path.join(data_path,str(rand_sub).zfill(3),str(subB_seq))
            
            assert os.path.isdir(subB_path)

            subA_path = os.path.join(data_path,str(sub).zfill(3),seq_name)
            assert os.path.isdir(subA_path)

            if sub == sub_v and split =='test':
                self.sequences.append((subA_path,label,apex))
            elif sub != sub_v and split =='train':
                self.sequences.append((subA_path,label,apex,subA_neutral, subB_path, subB_neutral))
            else:
                continue;
           
        
    
    def __getitem__(self, index):
        if self.split == 'test':
            path, label, apex= self.sequences[index]
            for root, dirs, files in os.walk(path):
                for name in files:
                    if str(apex)in name:
                        img_name = name
                    else:
                        continue;
            img_name = os.path.join(path,img_name)
            assert os.path.isfile(img_name)
                
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

            return img, label
        elif self.split == 'train':
            path_A, label, apex, neutral_A, path_B, neutral_B= self.sequences[index]
            for root, dirs, files in os.walk(path_A):
                for name in files:
                    if str(apex)in name:
                        apex = name
                    elif str(neutral_A)in name:
                        neutral_A = name
                    else:
                        continue;
            apex = os.path.join(path_A,apex)
            assert os.path.isfile(apex)
            neutral_A = os.path.join(path_A,neutral_A)
            assert os.path.isfile(neutral_A)
            for root, dirs, files in os.walk(path_B):
                for name in files:
                    if str(neutral_B)in name:
                        neutral_B = name
                        break;
                    else:
                        continue;
            neutral_B = os.path.join(path_B,neutral_B)
            assert os.path.isfile(neutral_B)



            
            normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                                std = [0.22803, 0.22145, 0.216989])  #standard of resnet 3d
            transform = transforms.Compose([transforms.Resize((112,112)),
                                            transforms.ToTensor(),
                                            #transforms.RandomErasing(p=0.5,scale=(0.2, 0.3)), 
                                            normalize
                                            ])
            with Image.open(os.path.join(path_A,apex)).convert("RGB") as img:
                if self.transform is None:
                    img = transform(img)
                else:
                    img = self.transform(img)
            
            with Image.open(os.path.join(path_A,neutral_A)).convert("RGB") as img_A:
                if self.transform is None:
                    img_A = transform(img_A)
                else:
                    img_A = self.transform(img_A)
            with Image.open(os.path.join(path_B,neutral_B)).convert("RGB") as img_B:
                if self.transform is None:
                    img_B = transform(img_B)
                else:
                    img_B = self.transform(img_B)

            label = torch.tensor(label,dtype = torch.long)
            return img, label, img_A, img_B

    
    def __len__(self):
        return len(self.sequences)



#sub_v = random.randint(1, 27)
#casme = casme_dataset_v2(data_path='/home/lynn/Dataset/CASME_extend/CASME_occ/down',label_path="/home/lynn/Dataset/CASME_extend/casme/CASME2-coding-20190701.xlsx",sub_v=sub_v,split='train')
#casme = samm_triplet_dataset(data_path='/home/lynn/Dataset/SAMM/samm/cropped_202208/',label_path="/home/lynn/Dataset/SAMM/SAMM.csv",sub_v=sub_v,split='train')

