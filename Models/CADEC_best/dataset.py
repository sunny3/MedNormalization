from copy import copy
import torch
from typing import List
from vectorization import ConceptVectorizer
import numpy as np

class MedNormDataset(torch.utils.data.Dataset):
    def __init__(self, X: List[str], y: List[str], CV: ConceptVectorizer, use_cuda = False):
        '''
        Входные данные: 
        X - список из фраз
        y - список из концептов
        CV - векторизатор для терминов медры с уже полученными вложениями
        '''
        assert hasattr(CV, 'thesaurus_embeddings'), 'У объекта ConceptVectorizer должен быть вызван метод fit_transform()'
        self.CV = CV
        self.phrases = np.array(X)
        self.codes = np.array(y)
        self.terms = np.array([self.CV.meddra_code_to_meddra_term[concept] for concept in self.codes])
        self.X, self.y = self._vectorization(X, y)
        if use_cuda:
            self.X = self.X.to('cuda') 
            self.y = self.y.to('cuda')
        
    def _vectorization(self, phrases_text, concepts):
        '''
        Векторизует датасет. 
        Фразам для нормализации будет соответствовать их представление для модели трансформера
        Концептам их one-hot форма из словаря ConceptVectorizer'а
        '''
        #Переводим в one_hot концепты
        vectorized_concepts = []
        for concept in concepts:
            vectorized_concepts.append(self.CV.meddra_code_to_one_hot_emb(concept))
        vectorized_concepts = torch.tensor(vectorized_concepts, dtype=torch.float32)
        #Переводим фразы в форму представления для модели-трансформера
        vectorized_phrases_text = self.CV.tokenizer(phrases_text, padding=True, truncation=True, return_tensors='pt')
        return vectorized_phrases_text, vectorized_concepts
        
    def __len__(self):
        return self.X['input_ids'].size()[0]

    def __getitem__(self, idx):
        if type(idx)==int:
            idx = torch.tensor(idx)
        sample = {'tokenized_phrases': {'input_ids': self.X['input_ids'][idx], 
                                        'token_type_ids': self.X['token_type_ids'][idx],
                                        'attention_mask': self.X['attention_mask'][idx]},
                  'one_hot_labels': self.y[idx],
                  'phrases': self.phrases[np.array(idx)],
                  'label_codes': self.codes[np.array(idx)],
                  'label_terms': self.terms[np.array(idx)]
                 }
        return sample