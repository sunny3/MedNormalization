from hyperopt import hp
# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval, STATUS_OK
from sklearn.metrics import f1_score
from models import CADEC_SoTa
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
from dataset import MedNormDataset
import os
import torch.optim as optim
import torch.nn as nn
import torch

class CADEC_SoTa_Optimizer:
    '''
    Пока что поиск оптимального batch_size и learning_rate
    Пример сетки для поиска гиперпараметров, подаваемой в оптимизатор: {'lr': (0.0001, 0.001), 'batch_size': (16, 32, 64)}
    epochs - на сколько эпох запускать сетку
    max_evals - сколько итераций поиска гиперпараметров проводить (сколько раз запускать сетку)
    verbose - печатать ли информацию
    '''
    def __init__(self, model: CADEC_SoTa,  
                 train_data: MedNormDataset, 
                 test_data: MedNormDataset, 
                 hyperparam_space: dict[str, tuple], 
                 max_evals=1000, 
                 epochs = 1, 
                 verbose=False, 
                 use_cuda=True):
        assert set(hyperparam_space.keys()) == {'lr', 'batch_size'}, \
        "Hyper param space must contain only 'lr' and 'batch_size' parameters"
        self.initial_model = model
        self.train_data = train_data
        self.test_data = test_data
        self.hyperparam_space = {'lr': hp.uniform('lr', *hyperparam_space['lr']),
                                  'batch_size': hp.choice('batch_size', hyperparam_space['batch_size'])}
        self.initial_hyperparam_space = hyperparam_space
        self.device = 'cuda' if use_cuda else 'cpu'
        self.epochs = epochs
        self.verbose = verbose
        self.max_evals = max_evals
        if self.verbose:
            self.counter=0
        
    def train_model(self, args)->dict:
        lr, batch_size = args['lr'], int(args['batch_size'])
        #выставляем одни и те же настройки для детерминированности
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        torch.use_deterministic_algorithms(mode=False)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        
        #каждый раз переинициалазируем объекты
        model = deepcopy(self.initial_model)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size,
                                          shuffle=False, num_workers=0)
        self.test_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler()
        
        model.train()
        
        
        for epoch in range(1, self.epochs+1):     
            for data in self.train_data_loader:

                inputs = data['tokenized_phrases']
                labels = data['one_hot_labels']

                optimizer.zero_grad()
                if self.device=='cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)['output']
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)['output']
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        model.eval()
        model_answers=[]
        real_answers=[]
        for data in self.test_data_loader:
            inputs = data['tokenized_phrases']
            with torch.no_grad():
                outputs_dict = model(inputs)
                #outputs_dict.label_concepless_tensors(score_treshold = 6.1977e-05)
                pred_meddra_code = self.test_data.CV.meddra_codes[outputs_dict['output'].argmax()]
            model_answers.append(pred_meddra_code)
            real_answers.append(data['label_codes'])
        
        
        f1 = f1_score(real_answers, model_answers, average='micro')
        if self.verbose:
            self.counter+=1
            print('%s. f1: %s, lr: %s, bs: %s'%(self.counter, np.round(f1, 4), np.round(lr,7), batch_size))
        return {'loss': -f1,
                'status': STATUS_OK}
    
    def optimize(self):
        self.best_hyperparams = fmin(
        fn=self.train_model, # Objective Function to optimize
        space=self.hyperparam_space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=self.max_evals # Number of optimization attempts
       )
        choice_index = self.best_hyperparams['batch_size']
        self.best_hyperparams['batch_size'] = self.initial_hyperparam_space['batch_size'][choice_index]
        print('Best hyperparams: \n')
        print(self.best_hyperparams)
        #для обнуления счетчика запусков
        if self.verbose:
            self.counter=0
        return self.best_hyperparams