import json
from models import CADEC_SoTa
from dataset import MedNormDataset
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process the training args.')

parser.add_argument('-p', '--file4pred', type=str, 
                     help='Path to the data file in sagnlp format for making MedDRA labels prediction', required=True)
parser.add_argument('-res', '--res_dir', type=str, 
                     help='Path to the directory for file with predicted MedDRA labels', required=True)
parser.add_argument('-model', '--model_dir', type=str, 
                     help='Path to the directory with saved model and vectorizer', required=True)

args = parser.parse_args()

pred_data_path = args.file4pred
pred_model_dir = args.model_dir
res_path = args.res_dir
    
#pred_data_path = './NER_pred_dir_data/ner_test_pred.json'
#pred_model_dir = './Model_weights/rubert_11072022_test/'
#res_path = './NER_pred_dir_data/ner_test_pred_with_codes.json'

with open(pred_data_path, 'r') as f:
    data_pred = json.load(f)

y_pred_texts = []
y_pred_codes = []
y_pred_idx = []
for pred_idx, pred_rev in enumerate(data_pred):
    for pred_ent_idx, pred_ent in pred_rev['entities'].items():
        if 'Disease:DisTypeIndication' in pred_ent['tag'] or 'ADR' in pred_ent['tag']:
            pass
        else:
            continue
        y_pred_texts.append(pred_ent['text'])
        #заглушка ввиде кода 10000002
        y_pred_codes.append('10000002')
        y_pred_idx.append({'pred_idx': pred_idx, 
                           'entity_id': pred_ent_idx})
        
#Loading model
net, CV = CADEC_SoTa.load_model(pred_model_dir, return_vectorizer=True)

#net = CADEC_SoTa(transformer_model_path, CV.thesaurus_embeddings)
#net.load_state_dict(weights)
#torch.save(net, './Model_weights/rubert_11072022_test/model.pt')
#raise ValueError

if CV.use_concept_less:
    CV.switch_to_regular_mode()

torch_ds = MedNormDataset(y_pred_texts, y_pred_codes, CV, use_cuda=CV.use_cuda)
dsloader = torch.utils.data.DataLoader(torch_ds, batch_size=1, shuffle=False)

#net = CADEC_SoTa(transformer_model_path, CV.thesaurus_embeddings)
#net.load_state_dict(torch.load(model_path))
#torch.load(model_dir)

#switch to conceptless
if not CV.use_concept_less:
    CV.switch_to_concepless_mode()
net.to('cuda')
net.eval()
model_codes=[]
model_terms=[]
print('Making predictions...')
with tqdm(dsloader, unit="batch") as eval_process:
    for data in eval_process:

        inputs = data['tokenized_phrases']

        with torch.no_grad():
            outputs_dict = net(inputs)
            outputs_dict.label_concepless_tensors(score_treshold = 6.1977e-05)
            pred_meddra_code = CV.meddra_codes[outputs_dict['output'].argmax()]
            pred_meddra_term = CV.meddra_code_to_meddra_term[pred_meddra_code]
        model_terms.append(pred_meddra_term)
        model_codes.append(pred_meddra_code)
        
#paste predicted codes in initial json
i=0
for meddra_code, meddra_term in zip(model_codes, model_terms):
    #find specific review
    curr_rev = data_pred[y_pred_idx[i]['pred_idx']]
    #find entity
    curr_rev['entities'][y_pred_idx[i]['entity_id']]['MedDRA_code'] = meddra_code
    curr_rev['entities'][y_pred_idx[i]['entity_id']]['MedDRA'] = meddra_term
    i+=1
    
with open(res_path, 'w') as f:
    json.dump(data_pred, f)