from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from os.path import dirname, realpath
import numpy as np

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
    def __init__(self, model_path, thesaurus_path):
        assert thesaurus_path.split('.')[-1] == 'asc'
        print('loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.meddra_codes = []
        self.meddra_code_to_meddra_term = {}
        self.meddra_term_to_meddra_code = {}
        with open(thesaurus_path, "r") as f:
            for meddra_line in f:
                meddra_line = meddra_line.split("$")
                self.meddra_codes.append(meddra_line[0])
                self.meddra_code_to_meddra_term[meddra_line[0]] = meddra_line[1]
                self.meddra_term_to_meddra_code[meddra_line[1]] = meddra_line[0]
                
    def meddra_code_to_one_hot_emb(self, meddra_code):
        one_hot_emb = [0]*len(self.meddra_codes)
        one_hot_emb[self.meddra_codes.index(meddra_code)] = 1
        return one_hot_emb
    
    def decode_vec_to_meddra_code(self, vec):
        '''
        vec - либо one hot форма meddra кода, 
        либо выход модели, индекс макс.которой принимается за индекс кода meddra'''
        if type(vec) != np.array:
            vec = np.array(vec)
        return self.meddra_codes[vec.argmax()]
    
    def decode_vec_to_meddra_term(self, vec):
        return self.meddra_code_to_meddra_term[self.decode_vec_to_meddra_code(vec)]
    
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_cls_token_emb(self, model_output):
        embeddings = model_output.last_hidden_state[:, 0, :]
        #embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings
    
    def fit_transform(self, mode='mean_pooling'):
        print('getting concept embeddings in %s mode...'%mode)
        self.vectorization_mode = mode
        #for meddra_code, meddra_term in tqdm(self.meddra_code_to_meddra_term.items()):
            # Tokenize sentences
        encoded_input = self.tokenizer(list(self.meddra_code_to_meddra_term.values()), padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        if mode=='mean_pooling':
            self.thesaurus_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        elif mode=='cls_token':
            self.thesaurus_embeddings = self.get_cls_token_emb(model_output)
        else:
            raise AssertionError('%s is not implemented'%mode)
        
