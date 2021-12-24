from torch import nn
from transformers import BertModel
from layers import *
from torchcrf import CRF

import torch


class EntityBert(nn.Module):
    def __init__(self, args):
        super(EntityBert, self).__init__()
        
        self.max_turns = args.max_turns
        self.hidden_size = args.hidden_size
        self.pooling = args.pooling
        
        self.bert = BertModel.from_pretrained(args.bert_type)
        self.bert.resize_token_embeddings(new_num_tokens = args.vocab_size)

        self.dropout = nn.Dropout(args.dropout)
        self.context_encoder = None
        
        if args.turn_type == 'multi':
            self.context_encoder = ContextEncoder(
                self.hidden_size,
                args.context_d_ff,
                args.context_num_heads,
                args.context_dropout,
                args.context_num_layers,
                args.max_turns, 
                args.p_dim, 
                args.device,
            )
            output_dim = args.hidden_size * 2
        elif args.turn_type == 'single':
            output_dim = args.hidden_size
        
        self.position_wise_ff = nn.Linear(output_dim, args.num_classes)
        self.crf = CRF(args.num_classes, batch_first=True)
        
    def forward(self, x, tags, pad_id, turns=None):
        if turns is not None:
            batch_size, num_contexts = x.shape[0], x.shape[1]
            bert_masks = self.make_bert_mask(x, pad_id)  # (B, T, L)
            e_masks = self.make_encoder_mask(turns, num_contexts)  # (B, T)
            
            x_flattened = x.view(batch_size * self.max_turns, -1)  # (B, T, L) => (B*T, L)
            bert_masks_flattened = bert_masks.view(batch_size * self.max_turns, -1)  # (B, T, L) => (B*T, L)

            output = self.bert(input_ids=x_flattened.long(), attention_mask=bert_masks_flattened)[0]  # (B*T, L, d_h)
            output = output.view(batch_size, self.max_turns, -1, self.hidden_size)  # (B*T, L, d_h) => (B, T, L, d_h)
            
            history_embs = self.embed_context(output)  # (B, T, d_h)         
            encoder_output = self.context_encoder(history_embs, e_masks.unsqueeze(1))  # (B, T, d_h)

            context_vec = encoder_output[torch.arange(encoder_output.shape[0]), turns]  # (B, d_h)
            output = output[torch.arange(output.shape[0]), turns]  # (B, L, d_h)
            seq_len = output.shape[1]
            output = torch.cat((output, context_vec.unsqueeze(1).repeat(1, seq_len,1)), dim=-1)  # (B, L, 2*d_h)
            
            x_masks = bert_masks[torch.arange(bert_masks.shape[0]), turns]  # (B, L)
        else:
            x_masks = self.make_bert_mask(x, pad_id)  # (B, L)
        
            output = self.bert(input_ids=x, attention_mask=x_masks)[0]  # (B, L, d_h)
            
        emissions = self.position_wise_ff(output)  # (B, L, C)
        
        log_likelihood, sequence_of_tags = self.crf(emissions, tags, mask=x_masks.bool(), reduction='mean'), self.crf.decode(emissions, mask=x_masks.bool())
        return log_likelihood, sequence_of_tags  # (), (B, L)
        
    def init_model(self):
        init_list = [self.dropout, self.position_wise_ff, self.crf]
        for module in init_list:
            for param in module.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
    
    def embed_context(self, bert_output):
        if self.pooling == 'cls':
            return bert_output[:, :, 0]  # (B, T, d_h)
        elif self.pooling == 'mean':
            return torch.mean(bert_output, dim=2)
        elif self.pooling == 'max':
            return torch.max(bert_output, dim=2).values
        
    def make_bert_mask(self, x, pad_id):
        bert_masks = (x != pad_id).float()  # (B, L)
        return bert_masks
    
    def make_encoder_mask(self, turns, num_contexts):
        batch_size = turns.shape[0]
        masks = torch.zeros((turns.shape[0], num_contexts), device=turns.device)
        masks[torch.arange(num_contexts, device=masks.device) < turns[..., None]] = 1.0
        
        return masks

    
class ContextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout, num_layers, max_turns, p_dim, device):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.max_turns = max_turns
        self.p_dim = p_dim
        self.device = device
        
        self.positional_encoder = PositionalEncoder(self.max_turns, self.p_dim, self.device)
        self.linear = nn.Linear(self.d_model+self.p_dim, self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.num_heads, self.dropout) for i in range(self.num_layers)])
        self.layer_norm = LayerNormalization(self.d_model)

    def forward(self, x, e_masks):
        x = self.positional_encoder(x, cal='concat')  # (B, T, d_h)
        x = self.linear(x)  # (B, T, d_h)
        for i in range(self.num_layers):
            x = self.layers[i](x, e_masks)

        return self.layer_norm(x)
