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
        sample = {'tokenized_phrases': {k: self.X[k][idx] for k in self.X.keys()},
                                        #{'input_ids': self.X['input_ids'][idx], 
                                        #'token_type_ids': self.X['token_type_ids'][idx],
                                        #'attention_mask': self.X['attention_mask'][idx]},
                  'one_hot_labels': self.y[idx],
                  'phrases': self.phrases[np.array(idx)],
                  'label_codes': self.codes[np.array(idx)],
                  'label_terms': self.terms[np.array(idx)]
                 }
        return sample
    
class MedNormContextDataset(MedNormDataset):
    def __init__(self, X: List[dict], y: List[str], CV: ConceptVectorizer, use_cuda = False):
        '''
        Входные данные: 
        X - список из предложений, предложение - словарь из ключей sentence и phrase_spans и phrase
        y - список из концептов
        CV - векторизатор для терминов медры с уже полученными вложениями
        '''
        assert hasattr(CV, 'thesaurus_embeddings'), 'У объекта ConceptVectorizer должен быть вызван метод fit_transform()'
        self.CV = CV
        self.sentences = [d['sentence'] for d in X]
        self.phrases = np.array([d['phrase'] for d in X])
        self.codes = np.array(y)
        self.terms = np.array([self.CV.meddra_code_to_meddra_term[concept] for concept in self.codes])
        
        self.sentences, self.pharses, self.X, self.y = self._vectorization(X, y)
        if use_cuda:
            self.X = self.X.to('cuda') 
            self.y = self.y.to('cuda')
        
        
    def _create_phrase_mask(self, offset_mapping, entity_spans):
        '''Разметка предполагается разрывной'''
        if type(entity_spans)==dict:
            entity_spans = [[ent_s['begin'], ent_s['end']] for ent_s in entity_spans]
        #Соберем все id токенов для каждого спана сущности
        entity_token_ids = []
        for entity_span in entity_spans:
            #может можно покрасивее, хз, взял у Вани
            for token_id, token_span in enumerate(offset_mapping):
                if int(token_span[0]) >= int(entity_span[0]) and int(token_span[1]) <= int(entity_span[1]):
                    entity_token_ids.append(token_id)
        #сделаем маску
        entity_mask = []
        for i in range(len(offset_mapping)):
            if i in entity_token_ids:
                entity_mask.append(1)
            else:
                entity_mask.append(0)
        return torch.tensor(entity_mask)
        
    def _vectorization(self, sentences_with_markup, concepts):
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
        sentences_text = [s['sentence'] for s in sentences_with_markup]
        sentences_markup = [s['phrase_spans'] for s in sentences_with_markup]
        vectorized_sentences_text = self.CV.tokenizer(sentences_text, padding=True, \
                                                      truncation=True, \
                                                      return_offsets_mapping=True, \
                                                      return_tensors='pt')
        input_phrases_masks = torch.stack(list(map(self._create_phrase_mask, \
                                                    vectorized_sentences_text['offset_mapping'], \
                                                    sentences_markup)))
        vectorized_sentences_text['input_phrases_masks'] = input_phrases_masks
        return (np.array(sentences_text), 
                np.array([s['phrase'] for s in sentences_with_markup]), 
                vectorized_sentences_text, 
                vectorized_concepts)