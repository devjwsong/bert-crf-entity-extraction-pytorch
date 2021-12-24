from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from entity_bert import EntityBert
from custom_dataset import CustomDataset
from transformers import BertConfig, BertTokenizer, get_polynomial_decay_schedule_with_warmup
from seqeval.metrics import accuracy_score, f1_score
from itertools import chain

import torch
import random
import os, sys
import numpy as np
import argparse
import time
import json


def run(args):
    # Device setting
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        print("CUDA is unavailable. Starting with CPU.")
        args.device = torch.device('cpu')

    print(f"{args.turn_type}-turn setting fixed.")
    if args.turn_type == 'multi':
        print(f"Pooling policy is {args.pooling}.")
        
    # Load class dictionary
    print("Loading the class dictionary...")
    with open(f"{args.data_dir}/{args.processed_dir}/class_dict.json", 'r') as f:
        class_dict = json.load(f)
    args.num_classes = len(class_dict)
    idx2class = {v:k for k, v in class_dict.items()}

    # Adding arguments
    bert_config = BertConfig().from_pretrained(args.bert_type)
    args.hidden_size = bert_config.hidden_size
    args.p_dim = args.hidden_size
    args.max_len = min(args.max_len, bert_config.max_position_embeddings)

    # Tokenizer
    print("Loading the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)
    num_new_tokens = tokenizer.add_special_tokens(
        {
            'additional_special_tokens': [args.sp1_token, args.sp2_token]
        }
    )
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)

    args.cls_token = tokenizer.cls_token
    args.sep_token = tokenizer.sep_token
    args.pad_token = tokenizer.pad_token

    args.cls_id = vocab[args.cls_token]
    args.sep_id = vocab[args.sep_token]
    args.pad_id = vocab[args.pad_token]
    args.sp1_id = vocab[args.sp1_token]
    args.sp2_id = vocab[args.sp2_token]
    args.o_id = class_dict['O']

    # Load model & optimizer
    print("Loading the model and optimizer...")
    set_seed(args.seed)
    model = EntityBert(args).to(args.device)
    model.init_model()
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    # Loading datasets & dataloaders
    print(f"Loading {args.turn_type}-turn data...")
    train_set = CustomDataset(args, vocab, class_dict, prefix='train')
    valid_set = CustomDataset(args, vocab, class_dict, prefix='valid')
    test_set = CustomDataset(args, vocab, class_dict, prefix='test')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    # Setting scheduler
    num_batches = len(train_loader)
    args.total_train_steps = args.num_epochs * num_batches
    args.warmup_steps = int(args.warmup_ratio * args.total_train_steps)
    sched = get_polynomial_decay_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_train_steps,
        power=2.0,
    )
    
    # Training
    set_seed(args.seed)
    best_ckpt_path = train(args, model, optim, sched, train_loader, valid_loader, idx2class)
    
    # Testing
    print("Testing the model...")
    _, test_acc, test_f1 = evaluate(args, model, test_loader, idx2class, ckpt_path=best_ckpt_path)
    
    print("<Test Result>")
    print(f"Test accuracy: {test_acc} || Test F1 score: {test_f1}")
    print("GOOD BYE.")

              
def train(args, model, optim, sched, train_loader, valid_loader, idx2class):
    print("Training starts.")
    best_f1 = 0.0
    patience, threshold = 0, 1e-4
    best_ckpt_path = None
    
    for epoch in range(1, args.num_epochs+1):
        model.train()

        print("#"*50 + f" Epoch: {epoch} " + "#"*50)
        train_losses, train_ys, train_outputs, train_lens = [], [], [], []
        for i, batch in enumerate(tqdm(train_loader)):
            batch_x, batch_y, batch_lens, batch_turns = batch

            if args.turn_type == 'single':
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                log_likelihood, outputs = model(batch_x, batch_y, args.pad_id)  # (), (B, L)
            elif args.turn_type == 'multi':
                batch_x, batch_y, batch_turns = \
                    batch_x.to(args.device), batch_y.to(args.device), batch_turns.to(args.device)
                log_likelihood, outputs = model(batch_x, batch_y, args.pad_id, turns=batch_turns)  # (), (B, L)

            loss = -1 * log_likelihood

            model.zero_grad()
            optim.zero_grad()

            loss.backward()
            optim.step()
            sched.step()
            
            train_losses.append(loss.detach())
            train_ys.append(batch_y.detach())
            train_outputs.append(outputs)
            train_lens.append(batch_lens)
        
        train_losses = [loss.item() for loss in train_losses]
        train_loss = np.mean(train_losses)
        train_preds, train_trues = [], []
        for i in range(len(train_ys)):
            pred_batch, true_batch, batch_lens = train_outputs[i], train_ys[i], train_lens[i]
            
            batch_lens = batch_lens.tolist()  # (B)
            true_batch = [batch[:batch_lens[b]] for b, batch in enumerate(true_batch.tolist())]
            
            assert len(pred_batch) == len(true_batch)
            train_preds += pred_batch
            train_trues += true_batch
            
        assert len(train_preds) == len(train_trues)
        for i in range(len(train_preds)):
            train_pred, train_true = train_preds[i], train_trues[i]
            train_pred = [idx2class[class_id] for class_id in train_pred]
            train_true = [idx2class[class_id] for class_id in train_true]
            
            train_preds[i] = train_pred
            train_trues[i] = train_true
            
        train_acc = accuracy_score(train_trues, train_preds)
        train_f1 = f1_score(train_trues, train_preds)

        print(f"Train loss: {train_loss} || Train accuracy: {train_acc} || Train F1 score: {train_f1}")
        
        print("Validation processing...")
        valid_loss, valid_acc, valid_f1 = evaluate(args, model, valid_loader, idx2class)
        
        if valid_f1 >= best_f1 + threshold:
            best_f1 = valid_f1
            patience = 0
            best_ckpt_path = f"{args.ckpt_dir}/ckpt_epoch={epoch}_train_f1={round(train_f1, 4)}_valid_f1={round(valid_f1, 4)}"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"***** Current best checkpoint is saved. *****")
        else:
            patience += 1
            print(f"The f1 score did not improve by {threshold}. Patience: {patience}")

        print(f"Best validtion f1 score: {best_f1}")
        print(f"Validation loss: {valid_loss} || Validation accuracy: {valid_acc} || Current validation F1 score: {valid_f1}")
        
        if patience == 3:
            print("Run out of patience. Abort!")
            break

    print("Training finished!")
    
    return best_ckpt_path
    
    
def evaluate(args, model, eval_loader, idx2class, ckpt_path=None):
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    
    model.eval()

    eval_losses, eval_ys, eval_outputs, eval_lens = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader)):
            batch_x, batch_y, batch_lens, batch_turns = batch

            if args.turn_type == 'single':
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
                log_likelihood, outputs = model(batch_x, batch_y, args.pad_id)  # (), (B, L)
            elif args.turn_type == 'multi':
                batch_x, batch_y, batch_turns = \
                    batch_x.to(args.device), batch_y.to(args.device), batch_turns.to(args.device)
                log_likelihood, outputs = model(batch_x, batch_y, args.pad_id, turns=batch_turns)  # (), (B, L)

            loss = -1 * log_likelihood

            eval_losses.append(loss.detach())
            eval_ys.append(batch_y.detach())
            eval_outputs.append(outputs)
            eval_lens.append(batch_lens)

        eval_losses = [loss.item() for loss in eval_losses]
        eval_loss = np.mean(eval_losses)
        eval_preds, eval_trues = [], []
        for i in range(len(eval_ys)):
            pred_batch, true_batch, batch_lens = eval_outputs[i], eval_ys[i], eval_lens[i]
            
            batch_lens = batch_lens.tolist()  # (B)
            true_batch = [batch[:batch_lens[b]] for b, batch in enumerate(true_batch.tolist())]

            assert len(pred_batch) == len(true_batch)
            eval_preds += pred_batch
            eval_trues += true_batch

        assert len(eval_preds) == len(eval_trues)
        for i in range(len(eval_preds)):
            eval_pred, eval_true = eval_preds[i], eval_trues[i]
            eval_pred = [idx2class[class_id] for class_id in eval_pred]
            eval_true = [idx2class[class_id] for class_id in eval_true]

            eval_preds[i] = eval_pred
            eval_trues[i] = eval_true

        eval_acc = accuracy_score(eval_trues, eval_preds)
        eval_f1 = f1_score(eval_trues, eval_preds)

        return eval_loss, eval_acc, eval_f1

        
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help="The random seed.")
    parser.add_argument('--turn_type', required=True, type=str, help="The turn type setting. (Single-turn or Multi-turn)")
    parser.add_argument('--bert_type', default="bert-base-uncased", type=str, help="The BERT type to load.")
    parser.add_argument('--pooling', default="cls", type=str, help="The pooling policy when using the multi-turn setting.")
    parser.add_argument('--data_dir', default="data", type=str, help="The parent data directory.")
    parser.add_argument('--processed_dir', default="processed", type=str, help="The directory which will contain the parsed data pickle files.")
    parser.add_argument('--ckpt_dir', default="saved_models", type=str, help="The path for saved checkpoints.")
    parser.add_argument('--gpu', default=0, type=int, help="The index of the GPU to use.")
    parser.add_argument('--sp1_token', default="[USR]", type=str, help="The speaker1(USER) token.")
    parser.add_argument('--sp2_token', default="[SYS]", type=str, help="The speaker2(SYSTEM) token.")
    parser.add_argument('--max_len', default=128, type=int, help="The max length of each utterance.")
    parser.add_argument('--max_turns', default=5, type=int, help="The maximum number of the dialogue history to be attended in the multi-turn setting.")
    parser.add_argument('--dropout', default=0.1, type=float, help="The dropout rate.")
    parser.add_argument('--context_d_ff', default=2048, type=int, help="The size of intermediate hidden states in the feed-forward layer.")
    parser.add_argument('--context_num_heads', default=8, type=int, help="The number of heads for Multi-head attention.")
    parser.add_argument('--context_dropout', default=0.1, type=float, help="The dropout rate for the context encoder.")
    parser.add_argument('--context_num_layers', default=2, type=int, help="The number of layers in the context encoder.")
    parser.add_argument('--learning_rate', default=5e-5, type=float, help="The initial learning rate.")
    parser.add_argument('--warmup_ratio', default=0.0, type=float, help="The ratio of warmup steps to the total training steps.")
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size.")
    parser.add_argument('--num_workers', default=0, type=int, help="The number of sub-processes for data loading.")
    parser.add_argument('--num_epochs', default=10, type=int, help="The number of training epochs.")
              
    args = parser.parse_args()
    
    assert args.turn_type == 'single' or args.turn_type == 'multi', print("Please specify a correct turn type, either 'single' or 'multi'.")
    assert args.bert_type in [
        "bert-base-uncased",
        "bert-base-cased",
        "bert-large-uncased",
        "bert-large-cased"
    ]
    assert args.pooling in ["cls", "mean", "max"]
    
    run(args)
