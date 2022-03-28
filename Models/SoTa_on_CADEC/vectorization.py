from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from os.path import dirname, realpath
import numpy as np
from transformers.utils import logging
from copy import copy
from tqdm import tqdm
import time
#from transformers.utils.logging import enable_progress_bar, disable_progress_bar
#def mean_pooling(model_output, attention_mask):
#    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
#sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-roberta-large')
#model = AutoModel.from_pretrained('sentence-transformers/nli-roberta-large')

# Tokenize sentences
#encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
#with torch.no_grad():
#    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
#sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

#print("Sentence embeddings:")
#print(sentence_embeddings)

class ConceptVectorizer:  
    def __init__(self, model_path_or_model_obj, thesaurus_path, use_concept_less=False, use_model = True, use_cuda=False):
        assert thesaurus_path.split('.')[-1] == 'asc'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_model_obj)
        if use_model:
            self.model = AutoModel.from_pretrained(model_path_or_model_obj)
        self.meddra_codes = []
        self.meddra_code_to_meddra_term = {}
        self.meddra_term_to_meddra_code = {}
        if use_concept_less:
            self.meddra_codes.append('CONCEPT_LESS')
            self.meddra_code_to_meddra_term['CONCEPT_LESS'] = 'CONCEPT_LESS'
            self.meddra_term_to_meddra_code['CONCEPT_LESS'] = 'CONCEPT_LESS'
        self.use_concept_less = use_concept_less
        self.use_cuda = use_cuda
        self.meddra_len = 0
        with open(thesaurus_path, "r") as f:
            for meddra_line in f:
                self.meddra_len+=1
                meddra_line = meddra_line.split("$")
                self.meddra_codes.append(meddra_line[0])
                self.meddra_code_to_meddra_term[meddra_line[0]] = meddra_line[1]
                self.meddra_term_to_meddra_code[meddra_line[1]] = meddra_line[0]
                #для дебага
                #if self.meddra_len==2:
                #    break
                
    def meddra_code_to_one_hot_emb(self, meddra_code):
        one_hot_emb = [0]*len(self.meddra_codes)
        one_hot_emb[self.meddra_codes.index(meddra_code)] = 1
        return one_hot_emb
    
    def switch_to_concepless_mode(self):
        self.use_concept_less = True
        self.meddra_codes = ['CONCEPT_LESS'] + self.meddra_codes
        temp_dict = copy(self.meddra_code_to_meddra_term)
        self.meddra_code_to_meddra_term = {'CONCEPT_LESS': 'CONCEPT_LESS'}
        for k, v in temp_dict.items():
            self.meddra_code_to_meddra_term[k] = v
        temp_dict = copy(self.meddra_term_to_meddra_code)
        self.meddra_term_to_meddra_code = {'CONCEPT_LESS': 'CONCEPT_LESS'}
        for k, v in temp_dict.items():
            self.meddra_term_to_meddra_code[k] = v
            
    def switch_to_regular_mode(self):
        self.use_concept_less = False
        self.meddra_codes.pop(0)
        self.meddra_term_to_meddra_code.pop('CONCEPT_LESS', None)
        self.meddra_code_to_meddra_term.pop('CONCEPT_LESS', None)
        
    def define_new_model(self, model_path_or_model_obj):
        if type(model_path_or_model_obj)==str:
            print('loading model...')
            self.model = AutoModel.from_pretrained(model_path_or_model_obj)
        else:
            self.model = model_path_or_model_obj
        print('new model loaded...')
        
    def decode_vec_to_meddra_code(self, vec):
        '''
        vec - либо one hot форма meddra кода, 
        либо выход модели, индекс макс.которой принимается за индекс кода meddra'''
        if type(vec) != np.array:
            vec = np.array(vec)
        return self.meddra_codes[vec.argmax()]
    
    def decode_vec_to_meddra_term(self, vec):
        return self.meddra_code_to_meddra_term[self.decode_vec_to_meddra_code(vec)]
    
    
    def mean_pooling(self, token_embeddings, attention_mask):
        # = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_cls_token_emb(self, model_output):
        embeddings = model_output[:, 0, :]
        #embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings
    
    def fit_transform(self, mode='mean_pooling'):
        print('getting concept embeddings in %s mode...'%mode)
        self.vectorization_mode = mode
        #for meddra_code, meddra_term in tqdm(self.meddra_code_to_meddra_term.items()):
            # Tokenize sentences
        concept_less_margin = 0 if not self.use_concept_less else 1
        encoded_input = self.tokenizer(list(self.meddra_code_to_meddra_term.values())[concept_less_margin:], padding=True, truncation=True, return_tensors='pt')
        if self.use_cuda:
            encoded_input = encoded_input.to('cuda')
            self.model.to('cuda')
            print('moving to cuda... device name %s'%encoded_input['input_ids'].device)
        # Compute token embeddings
        start_time = time.time()
        print('Compute embeddings...')
        with torch.no_grad():
            if self.use_cuda:
                all_encoded_concepts = []
                #кодируем по одной фразе во избежании ошибки cuda outofmemmory, и чтобы tqdm красиво отображал все
                for i in tqdm(range(self.meddra_len)): #self.meddra_len
                    all_encoded_concepts.append(self.model(**{k: encoded_input[k][i].unsqueeze(0) for k in encoded_input.keys()})[0][0])
                model_output = torch.stack(all_encoded_concepts)
            else:
                print('in cpu mode there is no progress bar')
                model_output = self.model(**encoded_input)[0]
        # Perform pooling. In this case, max pooling.
        print('Embedding aggregation...')
        if mode=='mean_pooling':
            self.thesaurus_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        elif mode=='cls_token':
            self.thesaurus_embeddings = self.get_cls_token_emb(model_output)
        else:
            raise AssertionError('%s is not implemented'%mode)
        print("Concept embeddings have computed in %s seconds" % (time.time() - start_time))
        
