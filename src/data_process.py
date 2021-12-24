from tqdm import tqdm
from glob import glob
from transformers import BertTokenizer

import argparse
import os
import random
import json
import pickle


def load_file(file, tokenizer):
    total_tokens = []
    total_tags = []
    
    with open(file, 'r') as f:
        data = json.load(f)
        
    for dial in tqdm(data):
        dial_tokens, dial_tags = [], []
        turns = dial['utterances']
        for turn in turns:
            sp = turn['speaker']
            text = turn['text']
            
            tokens = tokenizer.tokenize(text)
            entity_tags = ['O'] * len(tokens)
            
            if 'segments' in turn:
                segs = turn['segments']
                tokens, entity_tags = find_entities(tokens, segs, entity_tags, tokenizer)
            
            assert len(tokens) == len(entity_tags)
            
            dial_tokens.append([sp] + tokens)
            dial_tags.append(entity_tags)
            
        assert len(dial_tokens) == len(dial_tags)
            
        total_tokens.append(dial_tokens)
        total_tags.append(dial_tags)
    
    assert len(total_tokens) == len(total_tags)
            
    return total_tokens, total_tags  # (N, T, L), (N, T, L)
        
        
def find_entities(tokens, segs, entity_tags, tokenizer):
    entity_list = [(seg['text'], seg['annotations'][0]['name']) for seg in segs]
    checked = [False] * len(tokens)
    
    for entity in entity_list:
        value, tag = entity
        entity_tokens = tokenizer.tokenize(value)

        entity_tags, checked = find_sublist(tokens, entity_tokens, tag, entity_tags, checked)
        
    return tokens, entity_tags
        
    
def find_sublist(full, sub, tag, entity_tags, checked):
    for i, e in enumerate(full):
        if e == sub[0] and not checked[i]:
            cand = full[i:i+len(sub)]
            
            if cand == sub:
                checked[i] = True
                entity_tags[i] = f'B-{tag}'
                
                if f'B-{tag}' not in class_dict:
                    class_dict[f'B-{tag}'] = len(class_dict)
                    class_dict[f'I-{tag}'] = len(class_dict)
                    class_dict[f'E-{tag}'] = len(class_dict)
                
                if len(sub) > 1:
                    entity_tags[i+len(sub)-1] = f'E-{tag}'
                    entity_tags = [f'I-{tag}' if cur_tag == 'O' and (j>i and j<i+len(sub)) else cur_tag for j, cur_tag in enumerate(entity_tags)]
                    
    return entity_tags, checked


def count_utter_num(tokens):
    count = 0
    for dialogue in tokens:
        count += len(dialogue)
        
    return count

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="The random seed.")
    parser.add_argument('--data_dir', default="data", type=str, help="The parent data directory.")
    parser.add_argument('--raw_dir', default="raw", type=str, help="The directory which contains the raw data json files.")
    parser.add_argument('--save_dir', default="processed", type=str, help="The directory which will contain the parsed data pickle files.")
    parser.add_argument('--bert_type', default="bert-base-uncased", type=str, help="The BERT type to load.")
    parser.add_argument('--train_ratio', default=0.8, type=float, help="The ratio of train set to the number of each domain file.")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("Loading the tokenizer...")
    assert args.bert_type in [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "bert-large-cased"
    ]
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)
    
    total_train_tokens = []
    total_train_tags = []
    total_valid_tokens = []
    total_valid_tags = []
    total_test_tokens = []
    total_test_tags = []
    
    class_dict = {'O': 0}
    
    json_files = glob(f"{args.data_dir}/{args.raw_dir}/*.json")
    print(json_files)
    
    for file in json_files:
        print(f"Processing {file}...")
        tokens, tags = load_file(file, tokenizer)
        
        pairs = list(zip(tokens, tags))
        random.shuffle(pairs)
        tokens, tags = list(zip(*pairs))
        tokens, tags = list(tokens), list(tags)
        
        train_tokens, train_tags = tokens[:int(len(tokens)*args.train_ratio)], tags[:int(len(tags)*args.train_ratio)]
        remained_tokens, remained_tags = tokens[int(len(tokens)*args.train_ratio):], tags[int(len(tags)*args.train_ratio):]
        
        valid_tokens, valid_tags = remained_tokens[:int(len(remained_tokens)*0.5)], remained_tags[:int(len(remained_tags)*0.5)]
        test_tokens, test_tags = remained_tokens[int(len(remained_tokens)*0.5):], remained_tags[int(len(remained_tags)*0.5):]
        
        total_train_tokens += train_tokens
        total_train_tags += train_tags
        total_valid_tokens += valid_tokens
        total_valid_tags += valid_tags
        total_test_tokens += test_tokens
        total_test_tags += test_tags
            
    train_utter_num = count_utter_num(total_train_tokens)
    valid_utter_num = count_utter_num(total_valid_tokens)
    test_utter_num = count_utter_num(total_test_tokens)
            
    save_dir = f"{args.data_dir}/{args.save_dir}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    print("Making the label dictionary...")
    with open(f"{save_dir}/class_dict.json", 'w') as f:
        json.dump(class_dict, f)
        
    print("Saving each data files...")
    with open(f"{save_dir}/train_tokens.pkl", 'wb') as f:
        pickle.dump(total_train_tokens, f)
    with open(f"{save_dir}/train_tags.pkl", 'wb') as f:
        pickle.dump(total_train_tags, f)
    with open(f"{save_dir}/valid_tokens.pkl", 'wb') as f:
        pickle.dump(total_valid_tokens, f)
    with open(f"{save_dir}/valid_tags.pkl", 'wb') as f:
        pickle.dump(total_valid_tags, f)
    with open(f"{save_dir}/test_tokens.pkl", 'wb') as f:
        pickle.dump(total_test_tokens, f)
    with open(f"{save_dir}/test_tags.pkl", 'wb') as f:
        pickle.dump(total_test_tags, f)
    
    print("Finished!")
    
    print("######################################## Anaysis ########################################")
    print(f"The number of train dialogues: {len(total_train_tokens)}")
    print(f"The number of valid dialogues: {len(total_valid_tokens)}")
    print(f"The number of test dialogues: {len(total_test_tokens)}")
    print(f"The number of train utterances: {train_utter_num}")
    print(f"The number of valid utterances: {valid_utter_num}")
    print(f"The number of test utterances: {test_utter_num}")
    