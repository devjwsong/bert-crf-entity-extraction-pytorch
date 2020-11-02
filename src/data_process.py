from tqdm import tqdm
from transformers import *

import argparse
import os
import json


def load_file(full_path, tokenizer):
    total_tokens = []
    total_tags = []
    
    with open(full_path, 'r') as f:
        lines = f.readlines()
        
    dialogue_tokens = []
    dialogue_tags = []
    for i, line in enumerate(tqdm(lines)):
        if line.strip() != config['dialogue_split']:
            comps = line.strip().split('\t')
            speaker = comps[0]
            sent = comps[1]
            
            tokens = tokenizer.tokenize(sent)
            entity_tags = ['O'] * len(tokens)
            
            if len(comps) > 2:
                entities = comps[2]
                tokens, entity_tags = find_entities(tokens, entities, entity_tags, tokenizer)
            
            dialogue_tokens.append(['[' + speaker + ']'] + tokens)
            dialogue_tags.append(entity_tags)
        else:
            total_tokens.append(dialogue_tokens)
            total_tags.append(dialogue_tags)
            dialogue_tokens = []
            dialogue_tags = []
            
    return total_tokens, total_tags  # (N, T, L), (N, T, L)
        
        
def find_entities(tokens, entities, entity_tags, tokenizer):
    entity_list = entities.split(config['outer_split_symbol'])
    checked = [False] * len(tokens)
    
    for entity in entity_list:
        entity_name = entity.split(config['inner_split_symbol'])[0]
        entity_tag = entity.split(config['inner_split_symbol'])[1]

        entity_tokens = tokenizer.tokenize(entity_name)

        entity_tags, checked = find_sublist(tokens, entity_tokens, entity_tag, entity_tags, checked)
        
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
            

def save_text(full_dir, dialogues, dialogue_labels):
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)
    
    with open(f"{full_dir}/inputs.txt", 'w') as f:
        for d, dialogue_tokens in enumerate(tqdm(dialogues)):
            history = []
            for t, tokens in enumerate(dialogue_tokens):
                speaker = tokens[0]
                sent = ' '.join(tokens[1:])
                utter = f"{speaker}\t{sent}"
                
                if t < config['max_time']:
                    history.append(utter)
                else:
                    history = history[1:] + [utter]
                
                f.write(f"{config['utter_split'].join(history)}\n")
                
    with open(f"{full_dir}/labels.txt", 'w') as f:
        for d, dialogue_tags in enumerate(tqdm(dialogue_labels)):
            for t, tags in enumerate(dialogue_tags):
                label = ' '.join(tags)
                f.write(f"{label}\n")
                

def save_ids(full_dir, dialogues, dialogue_labels):
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)
    
    with open(f"{full_dir}/inputs.ids", 'w') as f:
        for d, dialogue_tokens in enumerate(tqdm(dialogues)):
            history = []
            for t, tokens in enumerate(dialogue_tokens):
                speaker = tokens[0]
                sent = [str(tokenizer._convert_token_to_id(token)) for token in tokens[1:]]
                sent = ' '.join(sent)
                utter = f"{speaker}\t{sent}"
                
                if t < config['max_time']:
                    history.append(utter)
                else:
                    history = history[1:] + [utter]
                
                f.write(f"{config['utter_split'].join(history)}\n")

    with open(f"{full_dir}/labels.ids", 'w') as f:
        for d, dialogue_tags in enumerate(tqdm(dialogue_labels)):
            for t, tags in enumerate(dialogue_tags):         
                label = [str(class_dict[tag]) for tag in tags]
                label = ' '.join(label)
                f.write(f"{label}\n")


def count_utter_num(tokens):
    count = 0
    for dialogue in tokens:
        count += len(dialogue)
        
    return count

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, help="The path to configuration file.")
    
    args = parser.parse_args()
    
    print("Loading the configurations...")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['bert_name'])
    
    total_train_tokens = []
    total_train_tags = []
    total_valid_tokens = []
    total_valid_tags = []
    total_test_tokens = []
    total_test_tags = []
    
    class_dict = {'O': 0}
    
    file_list = os.listdir(f"{config['data_dir']}/{config['original_dir']}")
    for file in file_list:
        if os.path.isfile(f"{config['data_dir']}/{config['original_dir']}/{file}"):
            print(f"Processing {file}...")
            tokens, tags = load_file(f"{config['data_dir']}/{config['original_dir']}/{file}", tokenizer)
            train_tokens, train_tags = tokens[:int(len(tokens)*config['train_frac'])], tags[:int(len(tags)*config['train_frac'])]
            remained_tokens, remained_tags = tokens[int(len(tokens)*config['train_frac']):], tags[int(len(tags)*config['train_frac']):]
            
            f = config['valid_frac'] / (1.0-config['train_frac'])     
            valid_tokens, valid_tags = remained_tokens[:int(len(remained_tokens)*f)], remained_tags[:int(len(remained_tags)*f)]
            test_tokens, test_tags = remained_tokens[int(len(remained_tokens)*f):], remained_tags[int(len(remained_tags)*f):]

            total_train_tokens += train_tokens
            total_train_tags += train_tags
            total_valid_tokens += valid_tokens
            total_valid_tags += valid_tags
            total_test_tokens += test_tokens
            total_test_tags += test_tags
            
    train_utter_num = count_utter_num(total_train_tokens)
    valid_utter_num = count_utter_num(total_valid_tokens)
    test_utter_num = count_utter_num(total_test_tokens)
            
    full_dir = f"{config['data_dir']}/{config['entity_dir']}"
    if not os.path.isdir(full_dir):
        os.mkdir(full_dir)
    
    print("Making the label dictionary...")
    with open(f"{full_dir}/{config['tags_name']}.json", 'w') as f:
        json.dump(class_dict, f)
        
    print("Saving each data files...")
    save_text(f"{full_dir}/{config['train_name']}", total_train_tokens, total_train_tags)
    save_ids(f"{full_dir}/{config['train_name']}", total_train_tokens, total_train_tags)
    save_text(f"{full_dir}/{config['valid_name']}", total_valid_tokens, total_valid_tags)
    save_ids(f"{full_dir}/{config['valid_name']}", total_valid_tokens, total_valid_tags)
    save_text(f"{full_dir}/{config['test_name']}", total_test_tokens, total_test_tags)
    save_ids(f"{full_dir}/{config['test_name']}", total_test_tokens, total_test_tags)
    
    print("Finished!")
    
    print("######################################## Anaysis ########################################")
    print(f"The number of train dialogues: {len(total_train_tokens)}")
    print(f"The number of valid dialogues: {len(total_valid_tokens)}")
    print(f"The number of test dialogues: {len(total_test_tokens)}")
    print(f"The number of train utterances: {train_utter_num}")
    print(f"The number of valid utterances: {valid_utter_num}")
    print(f"The number of test utterances: {test_utter_num}")
    