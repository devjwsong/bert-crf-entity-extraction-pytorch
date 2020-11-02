from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer, config, class_dict):        
        input_path = f"{data_dir}/inputs.ids"
        label_path = f"{data_dir}/labels.ids"
        
        self.cls_id = tokenizer._convert_token_to_id(config['cls_token'])
        self.sep_id = tokenizer._convert_token_to_id(config['sep_token'])
        self.pad_id = tokenizer._convert_token_to_id(config['pad_token'])
        self.speaker1_id = tokenizer._convert_token_to_id(config['speaker1_token'])
        self.speaker2_id = tokenizer._convert_token_to_id(config['speaker2_token'])
        self.speaker1_token = config['speaker1_token']
        self.speaker2_token = config['speaker2_token']
        self.o_id = class_dict[config['o_tag']]
        
        self.utter_split = config['utter_split']
        self.max_len = config['max_len']
        self.max_time = config['max_time']
        
        self.tokenizer = tokenizer
        self.class_dict = class_dict
        
        with open(input_path, 'r') as f:
            input_lines = f.readlines()
            
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
            
        self.x = []
        self.y = []
        self.lens = []
        self.times = []
        
        if config['turn_type'] == 'single':
            self.process_single_turn(input_lines, label_lines)
        elif config['turn_type'] == 'multi':
            self.process_multi_turn(input_lines, label_lines)
            
        self.x = torch.LongTensor(self.x)  # (N, L) or (N, T, L)
        self.y = torch.LongTensor(self.y)  # (N, L)
        self.lens = torch.LongTensor(self.lens)  # (N)
        self.times = torch.LongTensor(self.times)  # (N)

    def process_single_turn(self, input_lines, label_lines):
        for i, input_line in enumerate(tqdm(input_lines)):
            utter = input_line.split(self.utter_split)[-1] 
            tags = label_lines[i].split(' ')  # (L)
            tags = [int(tag) for tag in tags] # (L)
    
            speaker = utter.split('\t')[0]
            tokens = utter.split('\t')[1].split(' ')
            tokens = [int(token) for token in tokens]
            
            if speaker == self.speaker1_token:
                speaker_id = self.speaker1_id
            elif speaker == self.speaker2_token:
                speaker_id = self.speaker2_id
                
            tokens, tags, valid_len = self.pad_or_truncate(tokens, speaker_id, tags)
            
            self.x.append(tokens)
            self.y.append(tags)
            self.lens.append(valid_len)
            self.times.append(0)
    
    def process_multi_turn(self, input_lines, label_lines):
        for i, input_line in enumerate(tqdm(input_lines)):
            utters = input_line.split(self.utter_split)  # (T)
            tags = label_lines[i].split(' ')  # (L)
            tags = [int(tag) for tag in tags] # (L)

            init_sent = [self.cls_id, self.sep_id] + [self.pad_id] * (self.max_len-2)
            history = [init_sent for i in range(self.max_time)]

            valid_len = -1
            time = len(utters)-1
            for t, utter in enumerate(utters):
                speaker = utter.split('\t')[0]
                tokens = utter.split('\t')[1].split(' ')
                tokens = [int(token) for token in tokens]

                if speaker == self.speaker1_token:
                    speaker_id = self.speaker1_id
                elif speaker == self.speaker2_token:
                    speaker_id = self.speaker2_id

                if t == time:
                    tokens, tags, valid_len = self.pad_or_truncate(tokens, speaker_id, tags)
                    history[t] = tokens
                else:
                    tokens, _, _ = self.pad_or_truncate(tokens, speaker_id)
                    history[t] = tokens

            self.x.append(history)
            self.y.append(tags)
            self.lens.append(valid_len)
            self.times.append(time)
    
    def pad_or_truncate(self, tokens, speaker_id, tags=None):
        tokens = [self.cls_id] + tokens + [self.sep_id, speaker_id, self.sep_id]
        
        if len(tokens) <= self.max_len:
            pad_len = self.max_len - len(tokens)
            tokens += ([self.pad_id] * pad_len)
            
            valid_len = -1
            if tags is not None:
                tags = [self.o_id] + tags + [self.o_id, self.o_id, self.o_id]
                valid_len = len(tags)
                tags += ([self.o_id] * pad_len)
        else:
            tokens = tokens[:self.max_len]
            tokens[-1] = self.sep_id
            tokens[-2] = speaker_id
            tokens[-3] = self.sep_id
            
            valid_len = -1
            if tags is not None:
                tags = [self.o_id] + tags + [self.o_id, self.o_id, self.o_id]
                tags = tags[:self.max_len]
                tags[-1] = self.o_id
                tags[-2] = self.o_id
                tags[-3] = self.o_id
                valid_len = self.max_len
            
        return tokens, tags, valid_len
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):        
        return self.x[idx], self.y[idx], self.lens[idx], self.times[idx]
