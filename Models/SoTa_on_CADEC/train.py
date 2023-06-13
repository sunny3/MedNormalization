from vectorization import ConceptVectorizer
import os
import torch

os.chdir('/s/ls4/users/romanrybka/pharm_er/Pipeline_Ner_Norm/RelationExtraction/src/normalization')

from tools.parse_RDRS import *
import numpy as np
import json

from tqdm import trange
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from evaluator import Evaluator
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tools.pytorchtools import EarlyStopping
import pandas as pd
from torch.optim.lr_scheduler import CyclicLR
from dataset import *
from models import CADEC_SoTa
import argparse

class ArgumentError(Exception):
    pass

#reqired
parser = argparse.ArgumentParser(description='Process the training args.')

parser.add_argument('-tr', '--train_data_path', type=str,
                     help='Path to the train data file in sagnlpjson format', required=True)
parser.add_argument('-res', '--res_dir', type=str,
                     help='Path to the directory for trained model and saved vectorizer', required=True)
parser.add_argument('-args', '--train_args', type=str,
                     help='Path to the configuration file, which have lr, epochs, batch size, use_cuda', default='./train_args.txt')

#other
parser.add_argument('-model', '--transformer_model_path', type=str,
                     help='Path to the initial transformer model, listed in https://huggingface.co/, which will be fine-tuned', default='DeepPavlov/rubert-base-cased')
parser.add_argument('-dict', '--meddra_path', type=str,
                     help='Path to the MedDRA file in asc. format, for example pt.asc or llt.asc. It defines what dictionary you will use', default='./Demo/mkb_path/mkb.asc')
parser.add_argument('-val', '--validation_data_path', type=str,
                     help='Path to the validation data file in sagnlpjson format to use early stopping and compute metrics')
parser.add_argument('-load_vect', '--saved_vectorizer_dir', type=str,
                     help='Path to the directory with configured vectorizer, when you define it, you dont need to provide "meddra_path"')
parser.add_argument('-load_pretrained', '--pretrained_model_dir', type=str,
                     help='Path to already fine-tuned model for futher finetuning (training) with saved vectorizer, when you define it, you dont need to provide "transformer_model_path"')

#for evaluate
parser.add_argument('-ts', '--test_data_path', type=str,
                     help='Path to the test data file in sagnlpjson format to evaluate after training')
parser.add_argument('--use_conceptless', action='store_true')

parser.add_argument('--use_cuda', action='store_true')

args = parser.parse_args()

#print(args)


train_data_path = args.train_data_path
transformer_model_path = args.transformer_model_path
res_dir = args.res_dir

if not os.path.exists(args.train_data_path):
    raise OsError("Train path doesn't exist")

if 'meddra_path' in args.__dict__:
    meddra_path = args.meddra_path
else:
    meddra_path = None

if 'transformer_model_path' in args.__dict__:
    transformer_model_path = args.transformer_model_path
else:
    transformer_model_path = None

if 'validation_data_path' in args.__dict__:
    validation_data_path = args.validation_data_path
else:
    validation_data_path = None

#if 'saved_vectorizer_dir' in args.__dict__:
#    saved_vectorizer_dir = args.saved_vectorizer_dir
#else:
#    saved_vectorizer_dir = None

if 'pretrained_model_dir' in args.__dict__:
    pretrained_model_dir = args.pretrained_model_dir
else:
    pretrained_model_dir = None

if 'test_data_path' in args.__dict__:
    test_data_path = args.test_data_path
else:
    test_data_path = None

if not pretrained_model_dir and not transformer_model_path:
    raise ArgumentError('Neither "pretrained_model_dir" nor "transformer_model_path" provided')
if not meddra_path and not saved_vectorizer_dir:
    raise ArgumentError('Neither "meddra_path" nor "saved_vectorizer_dir" provided')


USE_CUDA = args.use_cuda
config_file = args.train_args

#initialize objects
if pretrained_model_dir:
    print('Loading ConceptVectorizer\n---------')
    CV = ConceptVectorizer.load_vectorizer(pretrained_model_dir)
    print('loaded')

elif args.saved_vectorizer_dir:
    print('Loading ConceptVectorizer\n---------')
    CV = ConceptVectorizer.load_vectorizer(args.saved_vectorizer_dir)
    print('loaded')
else:
    print('Initializing ConceptVectorizer\n---------')
    use_model = True
    CV = ConceptVectorizer(transformer_model_path, meddra_path, \
                           use_concept_less=False, use_model=use_model, use_cuda=USE_CUDA)
    CV.fit_transform(mode='mean_pooling')
    CV.save_vectorizer(res_dir)

with open(train_data_path, 'r') as f:
    train_data = json.load(f)
    train_phrases, train_concepts = simple_parse_sagnlp_RDRS_mkb(train_data, CV)

if validation_data_path:
    with open(validation_data_path, 'r') as f:
        validation_data = json.load(f)
        val_phrases, val_concepts = simple_parse_sagnlp_RDRS_mkb(validation_data, CV)

if test_data_path:
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        filter_conceptless = not args.use_conceptless
        test_phrases, test_concepts = simple_parse_sagnlp_RDRS_mkb(test_data, CV)

ds_train = MedNormDataset(train_phrases, train_concepts, CV, use_cuda=USE_CUDA)
if validation_data_path:
    ds_val = MedNormDataset(val_phrases, val_concepts, CV, use_cuda=USE_CUDA)
if test_data_path:
    if args.use_conceptless:
        CV.switch_to_concepless_mode()
    ds_ts = MedNormDataset(test_phrases, test_concepts, CV, use_cuda=USE_CUDA)
    if args.use_conceptless:
        #we should then validate without concepless
        CV.switch_to_regular_mode()


train_config = {}
with open(config_file, 'r') as f:
    for line in f:
        if line=='\n':
            continue
        key, value = line.rstrip('\n').split(': ')
        train_config[key]=value

train_config['lr']=float(train_config['lr'])
train_config['epochs']=int(train_config['epochs'])
train_config['batch_size']=int(train_config['batch_size'])

train_config['transformer_model_path'] = transformer_model_path if not pretrained_model_dir else pretrained_model_dir
train_config['train_data_path'] = train_data_path
train_config['res_dir'] = res_dir


if pretrained_model_dir:
    print('Loading Model\n---------')
    net = CADEC_SoTa.load_model(pretrained_model_dir)
    print('loaded')
else:
    print('Initializing Model\n---------')
    net = CADEC_SoTa(transformer_model_path, CV.thesaurus_embeddings)
device = 'cuda' if USE_CUDA else 'cpu'
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=train_config['lr'])
scaler = torch.cuda.amp.GradScaler()
if validation_data_path:
    early_stopping = EarlyStopping(path=res_dir, mode='max', verbose=True)
trainloader = torch.utils.data.DataLoader(ds_train, batch_size=train_config['batch_size'],
                                          shuffle=False, num_workers=0)
if validation_data_path:
    valloader = torch.utils.data.DataLoader(ds_val, batch_size=1, shuffle=False)
    valid_evaluator = Evaluator(train_concepts, val_concepts)

if test_data_path:
    tsloader = torch.utils.data.DataLoader(ds_ts, batch_size=1, shuffle=False)
    test_evaluator = Evaluator(train_concepts, test_concepts)
#test_evaluator = Evaluator(train_concepts, test_concepts)

initial_loss = None

print('Config of the train process\n---------\n')
for k, v in train_config.items():
    print('%s: %s'%(k,v))
print('\n')
print('Train model')
for epoch in range(1, train_config['epochs']+1):
    #train model in a epoch
    with tqdm(trainloader, unit="batch") as tepoch:
        for data in tepoch:

            tepoch.set_description(f"Epoch {epoch}")

            inputs = data['tokenized_phrases']
            labels = data['one_hot_labels']

            optimizer.zero_grad()
            if USE_CUDA:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)['output']
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(inputs)['output']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()
            tepoch.set_postfix(loss_decrease = str(initial_loss/loss.item()))
    #calculations of metrics at the end of each epoch 
    if validation_data_path:
        net.eval()
        model_answers=[]

        with tqdm(valloader, unit="batch") as eval_process:
            for data in eval_process:

                inputs = data['tokenized_phrases']

                with torch.no_grad():
                    outputs_dict = net(inputs)
                    pred_meddra_code = CV.meddra_codes[outputs_dict['output'].argmax()]

                model_answers.append(pred_meddra_code)

        valid_evaluator.get_all_f1_scores(model_answers)
        early_stopping(valid_evaluator.f1_scores_df['All']['f1 macro']['score'], net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        net.train()

if test_data_path:
    print('Evaluating on a test dataset...')
    if validation_data_path:
        net = CADEC_SoTa.load_model(res_dir)
    if args.use_conceptless:
        CV.switch_to_concepless_mode()
    net.eval()
    model_answers=[]
    with tqdm(tsloader, unit="batch") as eval_process:
        for data in eval_process:

            inputs = data['tokenized_phrases']

            with torch.no_grad():
                outputs_dict = net(inputs)
                if args.use_conceptless:
                    outputs_dict.label_concepless_tensors(score_treshold = 6.1977e-05)
                pred_meddra_code = CV.meddra_codes[outputs_dict['output'].argmax()]
            model_answers.append(pred_meddra_code)

        test_evaluator.get_all_f1_scores(model_answers)


if not validation_data_path:
    print('Saving model with vectorizer...')
    net.save_model(res_dir)
else:
    print('Save only vectorizer, because best model have been already saved during validation')


print('Saved in %s'%res_dir)
#net.load_state_dict(torch.load(best_model_path))
