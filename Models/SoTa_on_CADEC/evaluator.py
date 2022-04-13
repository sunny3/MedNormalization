from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

class Evaluator:
    '''Расчет точностей и работа с индексами In and Out of VOC'''
    def __init__(self, train_conepts, test_concepts):
        self.train_conepts, self.test_concepts = np.array(train_conepts), np.array(test_concepts)
        self.get_in_out_of_voc_idx()
        
    def get_in_out_of_voc_idx(self):
        mask_out_of_voc = np.in1d(self.test_concepts, self.train_conepts) #маска элементов первого array, которые есть во втором array
        self.in_voc_idx = np.nonzero(mask_out_of_voc)[0] 
        self.out_of_voc_idx = np.nonzero(~mask_out_of_voc)[0]
        
    def get_f1_score(self, y_true, y_pred, average, idx='All'):
        if idx=='All':
            return f1_score(y_true, y_pred, average=average)
        elif idx=='In VOC':
            return f1_score(y_true[self.in_voc_idx], y_pred[self.in_voc_idx], average=average)
        elif idx=='Out of VOC':
            return f1_score(y_true[self.out_of_voc_idx], y_pred[self.out_of_voc_idx], average=average)
        
    def get_all_f1_scores(self, y_true, y_pred, pretty_print=True):
        columns_tuples = [('All', 'f1 micro'), 
                     ('All', 'f1 macro'),
                                           
                     ('In VOC', 'f1 micro'), 
                     ('In VOC', 'f1 macro'),
                                           
                     ('Out of VOC', 'f1 micro'), 
                     ('Out of VOC', 'f1 macro')]
        all_f1_scores = []
        for cfg_row in columns_tuples:
            all_f1_scores.append(self.get_f1_score(np.array(y_true), np.array(y_pred), 
                                                   average=cfg_row[1][3:], idx=cfg_row[0]))
        all_f1_scores = np.array(all_f1_scores).reshape((1,6))
        col_index = pd.MultiIndex.from_tuples(columns_tuples)
        self.f1_scores_df = pd.DataFrame(all_f1_scores, columns=col_index, index=['score'])
        if pretty_print:
            display(self.f1_scores_df) #, headers='keys', tablefmt='psql')