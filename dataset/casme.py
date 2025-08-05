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

class casme_dataset(data.Dataset):
    def __init__(self, data_path, label_path, sub_v, split, transform=None):
        ############## use only onset, apex and offset ##################
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        
        # load data_file to df
        df = pd.read_csv(self.label_path)#.drop(['Unnamed: 2','Unnamed: 6'], axis=1)
        
        # build emotion label dictionary 
        emotion_dic = {'others': 0, 'surprise': 1, 'happiness': 2, 'repression': 3, 'disgust': 4}
        
        # compute label weight
        value = torch.tensor(df.groupby("Label").count().sort_index().iloc[:, 0].values)
        self.weight = torch.div(torch.full_like(value, torch.max(value)), value)
        
        # build sequence of path 
        self.sequences = []
        for index,row in df.iterrows():
            sub = row['Subject']
            seq_name = row['Filename']
            label = row['Label']
            ### CASNE II Settings ###
            if label in ['fear','sadness']:
                continue;
            ### Apex Frames ###
            apex = row['Apex']
            if apex == '/':
                continue;
            
            label = emotion_dic[label]
            
            seq_path = os.path.join(data_path,'sub'+str(sub).zfill(2),seq_name)
            if sub == sub_v and split =='test':
                self.sequences.append((seq_path,label,apex))
            elif sub != sub_v and split =='train':
                self.sequences.append((seq_path,label,apex))
            else:
                continue;
        self.emotion_dic = emotion_dic             
        #print('CASME II dataset is built ! !')
    

    def __getitem__(self, index):
        path, label, apex= self.sequences[index]
        #img_name = 'img%d.jpg'%apex
        img_name = 'reg_img%d.jpg'%apex
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

        return img, label, path, img_name
    
    
    
    def __len__(self):
        return len(self.sequences)

class casme_surgical_dataset(data.Dataset):
    def __init__(self, data_path, label_path, sub_v, split, mask=None, transform=None):
        ############## use only onset, apex and offset ##################
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        if mask == None:
            # load data_file to df
            df = pd.read_csv(self.label_path)#.drop(['Unnamed: 2','Unnamed: 6'], axis=1)
            
            # build emotion label dictionary 
            emotion_dic = {'others': 0, 'surprise': 1, 'happiness': 2, 'repression': 3, 'disgust': 4}
                
            # build sequence of path 
            self.sequences = []
            for index,row in df.iterrows():
                sub = row['Subject']
                seq_name = row['Filename']
                label = row['Label']
                ### CASNE II Settings ###
                if label in ['fear','sadness']:
                    continue;
                ### Apex Frames ###
                apex = row['Apex']
                if apex == '/':
                    continue;
                
                label = emotion_dic[label]
                
                seq_path = os.path.join(data_path,'sub'+str(sub).zfill(2),seq_name)
                if sub == sub_v and split =='test':
                    self.sequences.append((seq_path,label,apex))
                elif sub != sub_v and split =='train':
                    self.sequences.append((seq_path,label,apex))
                else:
                    continue;
        self.emotion_dic = emotion_dic             
        
        #print('CASME II dataset is built ! !')

    def __getitem__(self, index):
        path, label, apex= self.sequences[index]
        img_name = 'reg_img%d_surgical.jpg'%apex
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

        return img, label#, path, img_name
    
    
    
    def __len__(self):
        return len(self.sequences)




'''
sub_v = random.randint(1, 27)
casme = casme_dataset_v2(data_path='/home/lynn/Dataset/CASME_extend/CASME_occ/down',label_path="/home/lynn/Dataset/CASME_extend/casme/CASME2-coding-20190701.xlsx",sub_v=sub_v,split='train')
'''
