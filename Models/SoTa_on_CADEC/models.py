import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoConfig, AutoModel
from collections import UserDict

class CADEC_SoTa_output(UserDict):
    def __init__(self, output):
        if type(output)==dict:
            super(CADEC_SoTa_output, self).__init__(output)
        else:
            super(CADEC_SoTa_output, self).__init__()
            self.data['output'] = output
            self.data['has_padding_for_conceptless_class'] = False
    def to(self, device: str):
        self.data = {k: v.to(device=device) for k, v in self.data.items()}
    def pad_output(self):
        #нулевой класс должен быть под индексом 1
        self.data['output'] = torch.cat((torch.zeros((*(self.data['output'].size()[:-1]), 1), \
                               device = self.data['output'].device), self.data['output']), dim=-1)
        self.data['has_padding_for_conceptless_class'] = True     
    def delete_padding(self):
        self.data['output'] = self.data['output'][:,1:]
        self.data['has_padding_for_conceptless_class'] = False        
    def compute_scores(self):
        self.data['max_scores'] = torch.max(self.data['output'], dim=-1)[0]
    def mask_conceptless(self, score_treshold):
        self.compute_scores()
        concept_exists_term_mask = self.data['max_scores'][:]>score_treshold
        concept_less_term_mask = ~concept_exists_term_mask
        return concept_less_term_mask
    def label_concepless_tensors(self, score_treshold):
        self.pad_output()
        concept_less_tensor = torch.zeros(self.data['output'].size()[-1], dtype=self.data['output'].dtype, device=self.data['output'].device)
        concept_less_tensor[0]=1
        self.data['output'][self.mask_conceptless(score_treshold)] = concept_less_tensor

class CADEC_SoTa(nn.Module):
    def __init__(self, model_path: str, thesaurus_embeddings: torch.tensor):
        super(CADEC_SoTa, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        self.thesaurus_len, self.hidden_state_size = thesaurus_embeddings.size()
        self.thesaurus_normalized_embs = nn.Parameter(self._normalize_embeddings(thesaurus_embeddings), requires_grad=False)
        
        
    def forward(self, x):
        transformer_inp = {k:v for k,v in x.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
        emb = self.transformer(**transformer_inp)
        term_mask = transformer_inp['attention_mask'] if 'input_phrases_masks' not in x.keys() else x['input_phrases_masks']
        x = self._mean_pooling(emb, term_mask)
        #имеем две матрицы x - (batch_size, emb_size) и thesaurus_embeddings - (thesaurus_size, emb_size)
        #надо посчитать косинусную близость близость между каждым вектором x и каждым вложением из тезауруса
        #решение: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
        x_n = x.norm(dim=1)[:, None] 
        x_n = x / torch.clamp(x_n, min=1e-8)
        cos_sim = torch.mm(x_n, self.thesaurus_normalized_embs)
        x = F.softmax(cos_sim, dim=1)
        return CADEC_SoTa_output(x)
    
    
    def _normalize_embeddings(self, emb):
        normalized_embs = emb.norm(dim=1)[:, None]
        normalized_embs = emb / torch.clamp(normalized_embs, min=1e-8)
        normalized_embs = normalized_embs.transpose(0, 1)
        return normalized_embs
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-8)
        
class CADEC_SoTa_context_v2(nn.Module):
    def __init__(self, model_path: str, thesaurus_embeddings: torch.tensor):
        super(CADEC_SoTa_context, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        
        self.thesaurus_len, self.thesaurus_hidden_state_size = thesaurus_embeddings.size()
        self.transformer_hidden_state_size = self.transformer.config.hidden_size
        self.linear = nn.Linear(self.transformer_hidden_state_size*2, self.thesaurus_hidden_state_size, bias=True)
        
        self.thesaurus_len, self.hidden_state_size = thesaurus_embeddings.size()
        self.thesaurus_normalized_embs = nn.Parameter(self._normalize_embeddings(thesaurus_embeddings), requires_grad=False)
        
        
    def forward(self, x):
        transformer_inp = {k:v for k,v in x.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']}
        emb = self.transformer(**transformer_inp)
        term_mask = x['input_phrases_masks']
        x = self._mean_pooling(emb, term_mask)
        #добавление контекста
        cls_emb = emb[0][:, 0, :]
        x = torch.concat((x, cls_emb), dim=1)
        x = self.linear(x)
        x = F.tanh(x)
        #имеем две матрицы x - (batch_size, emb_size) и thesaurus_embeddings - (thesaurus_size, emb_size)
        #надо посчитать косинусную близость близость между каждым вектором x и каждым вложением из тезауруса
        #решение: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
        x_n = x.norm(dim=1)[:, None] 
        x_n = x / torch.clamp(x_n, min=1e-8)
        cos_sim = torch.mm(x_n, self.thesaurus_normalized_embs)
        x = F.softmax(cos_sim, dim=1)
        return CADEC_SoTa_output(x)
    
    
    def _normalize_embeddings(self, emb):
        normalized_embs = emb.norm(dim=1)[:, None]
        normalized_embs = emb / torch.clamp(normalized_embs, min=1e-8)
        normalized_embs = normalized_embs.transpose(0, 1)
        return normalized_embs
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-8)

