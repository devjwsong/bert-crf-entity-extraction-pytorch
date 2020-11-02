from torch import nn
from transformers import *
from layers import *
from torchcrf import CRF

import torch
import numpy as np
import random


class EntityBert(nn.Module):
    def __init__(self, config):
        super(EntityBert, self).__init__()
        
        self.config = config
        
        # Random seed fixing
        np.random.seed(777)
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        random.seed(777)
        
        self.bert = BertModel.from_pretrained(self.config['bert_name'])
        self.bert.resize_token_embeddings(new_num_tokens = self.config['vocab_size'])

        self.dropout = nn.Dropout(self.config['dropout'])
        self.context_encoder = None
        
        if self.config['turn_type'] == 'multi':
            self.context_encoder = ContextEncoder(
                self.config['hidden_size'],
                self.config['context_d_ff'],
                self.config['context_num_heads'],
                self.config['context_dropout'],
                self.config['context_num_layers'],
                self.config['max_time'], 
                self.config['p_dim'], 
                self.config['device'],
            )
            output_dim = self.config['hidden_size'] * 2
        elif self.config['turn_type'] == 'single':
            output_dim = self.config['hidden_size']
        
        self.position_wise_ff = nn.Linear(output_dim, self.config['num_classes'])
        self.crf = CRF(self.config['num_classes'], batch_first=True)
        
    def forward(self, x, pad_id, tags, times=None):
        if times is not None:
            bert_mask = self.make_bert_mask(x, pad_id)  # (B, T, L)
            e_mask = self.make_encoder_mask(times)  # (B, T)
            
            batch_size = x.shape[0]
            x_flattened = x.view(batch_size * self.config['max_time'], -1)  # (B, T, L) => (B*T, L)
            bert_mask_flattened = bert_mask.view(batch_size * self.config['max_time'], -1)  # (B, T, L) => (B*T, L)

            output = self.bert(input_ids=x_flattened.long(), attention_mask=bert_mask_flattened)[0]  # (B*T, L, d_h)
            output = output.view(batch_size, self.config['max_time'], -1, self.config['hidden_size'])  # (B*T, L, d_h) => (B, T, L, d_h)
            
            history_embs = self.embed_context(output)  # (B, T, d_h)         
            encoder_output = self.context_encoder(history_embs, e_mask.unsqueeze(1))  # (B, T, d_h)

            context_vec = encoder_output[torch.arange(encoder_output.shape[0]), times]  # (B, d_h)
            output = output[torch.arange(output.shape[0]), times]  # (B, L, d_h)
            output = torch.cat((output, context_vec.unsqueeze(1).repeat(1,self.config['max_len'],1)), dim=-1)  # (B, L, 2*d_h)
            
            x_mask = bert_mask[torch.arange(bert_mask.shape[0]), times]  # (B, L)
        else:
            x_mask = self.make_bert_mask(x, pad_id)  # (B, L)
        
            output = self.bert(input_ids=x, attention_mask=x_mask)[0]  # (B, L, d_h)
            
        emissions = self.position_wise_ff(output)  # (B, L, C)
        
        log_likelihood, sequence_of_tags = self.crf(emissions, tags, mask=x_mask.bool(), reduction='mean'), self.crf.decode(emissions, mask=x_mask.bool())
        return log_likelihood, sequence_of_tags  # (), (B, L)
        
    def init_model(self):
        init_list = [self.dropout, self.position_wise_ff, self.crf]
        for module in init_list:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
    
    def embed_context(self, bert_output):
        if self.config['sentence_embedding'] == 'cls':
            return bert_output[:, :, 0]  # (B, T, d_h)
        elif self.config['sentence_embedding'] == 'mean':
            return torch.mean(bert_output, dim=2)
        elif self.config['sentence_embedding'] == 'max':
            return torch.max(bert_output, dim=2).values
        
    def make_bert_mask(self, x, pad_id):
        bert_mask = (x != pad_id).float()
        return bert_mask
    
    def make_encoder_mask(self, times):
        e_mask = [[1] * (time+1) + [0] * (self.config['max_time']-time-1) for time in times.tolist()]
        e_mask = torch.Tensor(e_mask).to(self.config['device'])
        return e_mask

    
class ContextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers, max_time, p_dim, device):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.max_time = max_time
        self.p_dim = p_dim
        self.device = device
        
        self.positional_encoder = PositionalEncoder(self.max_time, self.p_dim, self.device)
        self.linear = nn.Linear(self.d_model+self.p_dim, self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.num_heads, self.dropout) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_mask):
        x = self.positional_encoder(x, cal='concat')  # (B, T, d_h)
        x = self.linear(x)  # (B, T, d_h)
        for i in range(self.num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)
