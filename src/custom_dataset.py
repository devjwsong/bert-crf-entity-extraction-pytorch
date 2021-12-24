from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import numpy as np
import pickle


class CustomDataset(Dataset):
    def __init__(self, args, vocab, class_dict, prefix):        
        with open(f"{args.data_dir}/{args.processed_dir}/{prefix}_tokens.pkl", 'rb') as f:
            tokens = pickle.load(f)
            
        with open(f"{args.data_dir}/{args.processed_dir}/{prefix}_tags.pkl", 'rb') as f:
            tags = pickle.load(f)
            
        self.input_ids = []
        self.labels = []
        self.valid_lens = []
        self.turns = []
        
        if args.turn_type == 'single':
            self.process_single_turn(args, vocab, class_dict, tokens, tags)
        elif args.turn_type == 'multi':
            self.process_multi_turns(args, vocab, class_dict, tokens, tags)
        
        assert len(self.input_ids) == len(self.labels)
        assert len(self.input_ids) == len(self.valid_lens)
        assert len(self.input_ids) == len(self.turns)
        
        print(f"{len(self.input_ids)} samples are prepared for {prefix} set.")
        
        self.input_ids = torch.LongTensor(self.input_ids)  # (N, L) or (N, T, L)
        self.labels = torch.LongTensor(self.labels)  # (N, L)
        self.valid_lens = torch.LongTensor(self.valid_lens)  # (N)
        self.turns = torch.LongTensor(self.turns)  # (N)

    def process_single_turn(self, args, vocab, class_dict, tokens, tags):
        assert len(tokens) == len(tags)
        
        for d in tqdm(range(len(tokens))):
            dial_tokens, dial_tags = tokens[d], tags[d]
            assert len(dial_tokens) == len(dial_tags)
            
            for u in range(len(dial_tokens)):
                utter_tokens, utter_tags = dial_tokens[u], dial_tags[u]
                sp, utter_tokens = utter_tokens[0], utter_tokens[1:]
                assert len(utter_tokens) == len(utter_tags)
                
                token_ids = [vocab[token] for token in utter_tokens]
                tag_ids = [class_dict[tag] for tag in utter_tags]
                
                if sp == "USER":  # Speaker1: USER
                    sp_id = args.sp1_id
                    token_ids, tag_ids, valid_len = self.pad_or_truncate(args, sp_id, token_ids, tag_ids)                
            
                    self.input_ids.append(token_ids)
                    self.labels.append(tag_ids)
                    self.valid_lens.append(valid_len)
                    self.turns.append(0)
    
    def process_multi_turns(self, args, vocab, class_dict, tokens, tags):
        assert len(tokens) == len(tags)
        
        for d in tqdm(range(len(tokens))):
            dial_tokens, dial_tags = tokens[d], tags[d]
            assert len(dial_tokens) == len(dial_tags)
            
            token_hists, tag_hists, len_hists = [], [], []
            for u in range(len(dial_tokens)):
                utter_tokens, utter_tags = dial_tokens[u], dial_tags[u]
                sp, utter_tokens = utter_tokens[0], utter_tokens[1:]
                assert len(utter_tokens) == len(utter_tags)
                
                token_ids = [vocab[token] for token in utter_tokens]
                tag_ids = [class_dict[tag] for tag in utter_tags]
                
                if sp == "USER":  # Speaker1: USER
                    sp_id = args.sp1_id
                    token_ids, tag_ids, valid_len = self.pad_or_truncate(args, sp_id, token_ids, tag_ids)
                elif sp == "ASSISTANT":  # Speaker2: SYSTEM
                    sp_id = args.sp2_id
                    token_ids, tag_ids, valid_len = self.pad_or_truncate(args, sp_id, token_ids) 
                    
                token_hists.append(token_ids)
                tag_hists.append(tag_ids)
                len_hists.append(valid_len)
                
            assert len(token_hists) == len(tag_hists)
            assert len(tag_hists) == len(len_hists)
            
            init_ids = [args.cls_id] + [args.pad_id] * (args.max_len-2) + [args.sep_id]
            for u in range(len(token_hists)):
                token_ids, tag_ids, valid_len = token_hists[u], tag_hists[u], len_hists[u]
                if token_ids[1] == args.sp1_id:
                    token_hist = token_hists[max(u+1-args.max_turns, 0):u+1]
                    assert len(token_hist[-1]) == len(tag_ids)
                    assert len(token_hist) <= args.max_turns
                    self.turns.append(len(token_hist)-1)
                    token_hist += [init_ids] * (args.max_turns-len(token_hist))
                    assert len(token_hist) == args.max_turns
                    self.input_ids.append(token_hist)
                    self.labels.append(tag_ids)
                    self.valid_lens.append(valid_len)
    
    def pad_or_truncate(self, args, sp_id, token_ids, tag_ids=None):
        token_ids = [args.cls_id, sp_id] + token_ids + [args.sep_id]
        if len(token_ids) <= args.max_len:
            pad_len = args.max_len - len(token_ids)
            token_ids += ([args.pad_id] * pad_len)
            
            valid_len = -1
            if tag_ids is not None:
                tag_ids = [args.o_id, args.o_id] + tag_ids + [args.o_id]
                valid_len = len(tag_ids)
                tag_ids += ([args.o_id] * pad_len)
        else:
            token_ids = token_ids[:args.max_len]
            token_ids[-1] = args.sep_id
            
            valid_len = -1
            if tag_ids is not None:
                tag_ids = [args.o_id, args.o_id] + tag_ids + [args.o_id]
                tag_ids = tag_ids[:args.max_len]
                tag_ids[-1] = args.o_id
                valid_len = args.max_len
        
        assert len(token_ids) == args.max_len
        if tag_ids is not None:
            assert len(token_ids) == len(tag_ids)
            
        return token_ids, tag_ids, valid_len
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, idx):        
        return self.input_ids[idx], self.labels[idx], self.valid_lens[idx], self.turns[idx]
