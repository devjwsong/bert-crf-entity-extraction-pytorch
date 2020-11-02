from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from entity_bert import *
from custom_dataset import *
from transformers import *
from sklearn.metrics import *
from itertools import chain

import torch
import os, sys
import numpy as np
import argparse
import time
import copy
import json


class Manager():
    def __init__(self, mode, turn_type, sentence_embedding, config_path, ckpt_name=None):
        print("Setting the configurations...")
        with open(args.config_path, 'r') as f:
            self.config = json.load(f)
            
        if self.config['device'] == "cuda":
            self.config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif self.config['device'] == "cpu":
            self.config['device'] = torch.device('cpu')
            
        self.config['mode'] = mode
        self.config['turn_type'] = turn_type
        self.config['sentence_embedding'] = sentence_embedding

        bert_config = BertConfig().from_pretrained(self.config['bert_name'])
        self.config['hidden_size'] = bert_config.dim
        self.config['p_dim'] = self.config['hidden_size']
        
        # Tokenizer & Vocab
        print("Loading the tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_name'])
        num_new_tokens = self.tokenizer.add_special_tokens(
            {
                'additional_special_tokens': [self.config['speaker1_token'], self.config['speaker2_token']]
            }
        )
        self.config['vocab_size'] = len(self.tokenizer.get_vocab())
        
        # Load class dictionary
        print("Loading the class dictionary...")
        full_dir = f"{self.config['data_dir']}/{self.config['entity_dir']}"
        with open(f"{full_dir}/{self.config['tags_name']}.json", 'r') as f:
            class_dict = json.load(f)
        self.config['num_classes'] = len(class_dict)
        
        # Load model & optimizer
        print("Loading the model and optimizer...")
        self.model = EntityBert(self.config).to(self.config['device'])
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.best_f1 = 0.0
        
        if not os.path.exists(self.config['ckpt_dir']):
            os.mkdir(self.config['ckpt_dir'])
        
        if ckpt_name is not None:
            assert os.path.exists(f"{self.config['ckpt_dir']}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            checkpoint = torch.load(f"{self.config['ckpt_dir']}/{ckpt_name}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_f1 = checkpoint['f1']
        else:
            print("Initializing the model...")
            self.model.init_model()
            
        print(f"Loading {self.config['turn_type']}-turn data...")
        if mode == 'train':
            # Load train & valid dataset
            train_set = CustomDataset(f"{full_dir}/{self.config['train_name']}", self.tokenizer, self.config, class_dict)
            valid_set = CustomDataset(f"{full_dir}/{self.config['valid_name']}", self.tokenizer, self.config, class_dict)
            self.train_loader = DataLoader(train_set, shuffle=True, batch_size=self.config['batch_size'])
            self.valid_loader = DataLoader(valid_set, shuffle=True, batch_size=self.config['batch_size'])
        elif mode == 'test':
            # Load test dataset
            test_set = CustomDataset(f"{full_dir}/{self.config['test_name']}", self.tokenizer, self.config, class_dict)
            self.test_loader = DataLoader(test_set, shuffle=True, batch_size=self.config['batch_size'])
        
        print("Setting finished.")
              
    def train(self):
        print("Training starts.")
              
        for epoch in range(1, self.config['num_epochs']+1):
            self.model.train()
            
            print(f"#################### Epoch: {epoch} ####################")
            train_losses = []
            train_pred = []
            train_true = []
            for i, batch in enumerate(tqdm(self.train_loader)):
                batch_x, batch_y, batch_lens, batch_times = batch
                batch_times = batch_times.squeeze(-1)  # (B)
                batch_lens = batch_lens.squeeze(-1)  # (B)
                pad_id = self.tokenizer._convert_token_to_id(self.config['pad_token'])
                
                if self.config['turn_type'] == 'single':
                    batch_x, batch_y = batch_x.to(self.config['device']), batch_y.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y)  # (), (B, L)
                elif self.config['turn_type'] == 'multi':
                    batch_x, batch_y, batch_times = \
                        batch_x.to(self.config['device']), batch_y.to(self.config['device']), batch_times.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y, times=batch_times)  # (), (B, L)
                
                loss = -1 * log_likelihood
                
                self.model.zero_grad()
                self.optim.zero_grad()
                
                loss.backward()
                self.optim.step()
                
                train_losses.append(loss.item())
                pred_list = list(chain.from_iterable(tag_seq))
                train_pred += pred_list
                true_list = list(chain.from_iterable([sublist[:batch_lens.tolist()[b]] for b, sublist in enumerate(batch_y.tolist())]))
                train_true += true_list
                
                assert len(pred_list) == len(true_list), "Please check if the length of predictions and that of true labels are identical."
             
            mean_train_loss = np.mean(train_losses)
            train_acc = accuracy_score(train_true, train_pred)
            train_f1 = f1_score(train_true, train_pred, average='weighted')
            print(f"Train loss: {mean_train_loss} || Train accuracy: {train_acc} || Train F1 score: {train_f1}")
            
            valid_loss, valid_acc, valid_f1 = self.validation()
              
            if valid_f1 > self.best_f1:
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'f1': self.best_f1
                }
              
                torch.save(state_dict, f"{self.config['ckpt_dir']}/best_ckpt.tar")
                print(f"***** Current best checkpoint is saved. *****")
                self.best_f1 = valid_f1
              
            print(f"Best validtion f1 score: {self.best_f1}")
            print(f"Validation loss: {valid_loss} || Validation accuracy: {valid_acc} || Current validation F1 score: {valid_f1}")
              
        print("Training finished!")
    
    def validation(self):
        print("Validation processing...")
        self.model.eval()
              
        valid_losses = []
        valid_pred = []
        valid_true = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.valid_loader)):
                batch_x, batch_y, batch_lens, batch_times = batch
                batch_times = batch_times.squeeze(-1)  # (B)
                batch_lens = batch_lens.squeeze(-1)  # (B)
                pad_id = self.tokenizer._convert_token_to_id(self.config['pad_token'])
                
                if self.config['turn_type'] == 'single':
                    batch_x, batch_y = batch_x.to(self.config['device']), batch_y.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y)  # (), (B, L)
                elif self.config['turn_type'] == 'multi':
                    batch_x, batch_y, batch_times = \
                        batch_x.to(self.config['device']), batch_y.to(self.config['device']), batch_times.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y, times=batch_times)  # (), (B, L)
                
                loss = -1 * log_likelihood
                
                valid_losses.append(loss.item())
                pred_list = list(chain.from_iterable(tag_seq))
                valid_pred += pred_list
                true_list = list(chain.from_iterable([sublist[:batch_lens.tolist()[b]] for b, sublist in enumerate(batch_y.tolist())]))
                valid_true += true_list
                
                assert len(pred_list) == len(true_list), "Please check if the length of predictions and that of true labels are identical."
              
        mean_valid_loss = np.mean(valid_losses)
        valid_acc = accuracy_score(valid_true, valid_pred)
        valid_f1 = f1_score(valid_true, valid_pred, average='weighted')
              
        return mean_valid_loss, valid_acc, valid_f1
        
              
    def test(self):
        print("Testing starts.")
        self.model.eval()
        
        test_losses = []
        test_pred = []
        test_true = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                batch_x, batch_y, batch_lens, batch_times = batch
                batch_times = batch_times.squeeze(-1)  # (B)
                batch_lens = batch_lens.squeeze(-1)  # (B)
                pad_id = self.tokenizer._convert_token_to_id(self.config['pad_token'])
                
                if self.config['turn_type'] == 'single':
                    batch_x, batch_y = batch_x.to(self.config['device']), batch_y.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y)  # (), (B, L)
                elif self.config['turn_type'] == 'multi':
                    batch_x, batch_y, batch_times = \
                        batch_x.to(self.config['device']), batch_y.to(self.config['device']), batch_times.to(self.config['device'])
                    log_likelihood, tag_seq = self.model(batch_x, pad_id, batch_y, times=batch_times)  # (), (B, L)
                
                loss = -1 * log_likelihood
                
                test_losses.append(loss.item())
                pred_list = list(chain.from_iterable(tag_seq))
                test_pred += pred_list
                true_list = list(chain.from_iterable([sublist[:batch_lens.tolist()[b]] for b, sublist in enumerate(batch_y.tolist())]))
                test_true += true_list
                
                assert len(pred_list) == len(true_list), "Please check if the length of predictions and that of true labels are identical."
              
        mean_test_loss = np.mean(test_losses)
        test_acc = accuracy_score(test_true, test_pred)
        test_f1 = f1_score(test_true, test_pred, average='weighted')
              
        print("#################### Test results ####################")
        print(f"Test loss: {mean_test_loss} || Test accuracy: {test_acc} || Test F1 score: {test_f1}")
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help="The path to configuration file.")
    parser.add_argument('--mode', required=True, type=str, help="Train or test?")
    parser.add_argument('--turn_type', required=True, type=str, help="Single-turn or multi-turn?")
    parser.add_argument('--sentence_embedding', required=False, type=str, help="How to embed a sentence?")
    parser.add_argument('--ckpt_name', required=False, type=str, help="Best checkpoint file.")
              
    args = parser.parse_args()
    
    assert args.mode == 'train' or args.mode=='test', print("Please specify a correct mode name, 'train' or 'test'.")
    assert args.turn_type == 'single' or args.turn_type=='multi', print("Please specify a correct turn type, 'single' or 'multi'.")
    
    if args.turn_type == 'multi':
        assert args.sentence_embedding == 'cls' or args.sentence_embedding == 'mean' or args.sentence_embedding == 'max', print("Please specify a correct sentence embedding method among 'cls', 'mean', and 'max'.")
              
    if args.mode == 'train':
        if args.ckpt_name is not None:
            manager = Manager(args.mode, args.turn_type, args.sentence_embedding, args.config_path, ckpt_name=args.ckpt_name)
        else:
            manager = Manager(args.mode, args.turn_type, args.sentence_embedding, args.config_path)
              
        manager.train()
        
    elif args.mode == 'test':
        assert args.ckpt_name is not None, "Please specify the trained model checkpoint."
        
        manager = Manager(args.mode, args.turn_type, args.sentence_embedding, args.config_path, ckpt_name=args.ckpt_name)
        
        manager.test()
