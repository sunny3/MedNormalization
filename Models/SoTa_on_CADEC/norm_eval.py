
import argparse as ap
import logging
import pandas as pd
from typing import List, Tuple, Dict, Union
from sklearn.metrics import precision_recall_fscore_support as prfs
import json


def read_json(path_to_df: Union[str, dict],include_concept_less:bool=False,consider_entity_tags:bool=True):
    _entities = []
    # Checking what type the path_to_df variable is
    if type(path_to_df) == str:
        # if filename
        with open(path_to_df) as f:
            df = json.load(f)
    else:
        # if list of documents
        df = path_to_df
    for s in df:
        sample_entities = []
        for _, ent in s['entities'].items():
            try:
                ent['MedDRA']
            except:
                continue

            tag=None
            for tg in ent['tag']:
                if tg=='ADR' or tg=='Disease:DisTypeIndication':
                    tag=tg
                    break

            if not consider_entity_tags:
                tag='tags_not_considered'

            if tag:
                med_entity = 'CONCEPT_LESS' if not ent['MedDRA'] else ent['MedDRA'].split('|')[0] #на случай если классов несколько
                if not include_concept_less:
                    if med_entity!='CONCEPT_LESS':
                        sample = (';'.join([str(i['begin']) for i in ent['spans']]),
                                  ';'.join([str(i['end']) for i in ent['spans']]),
                                  med_entity,tag)
                        sample_entities.append(sample)
                else:
                    sample = (';'.join([str(i['begin']) for i in ent['spans']]),
                              ';'.join([str(i['end']) for i in ent['spans']]),
                              med_entity,tag)
                    sample_entities.append(sample)

        _entities.append(sample_entities)

    return _entities # format: [[('182', '192', 'Speech disorder')], [('488', '504', 'Prevention'), ('869', '885', 'Prevention')]]

def compute_scores(gt_ent, pred_ent, log=None, consider_entity_tags=False):
    print("Evaluation")

    print("")
    print("--- Entities (NER) ---")
    print("An entity is considered correct if entity tag and span is predicted correctly")
    print("")
    gt_ner, pred_ner = _convert_by_setting(gt_ent, pred_ent, include_entity_types=False) 
    gt_flat, pred_flat, types = _score(gt_ner, pred_ner) # 1d massive, for f1 metric
    ner_eval = _compute_metrics(gt_flat, pred_flat, types, print_results=True,save_results=False) # eval with f1 score

    print("")
    print("--- Entities (NER+Norm) ---")
    print("An entity is considered correct if entity tag, concept and span is predicted correctly")
    print("")

    if (consider_entity_tags):
        label_ind=3
    else:
        label_ind=2
    gt_flat, pred_flat, types = _score(gt_ent, pred_ent,label_ind=label_ind) 
    _compute_metrics(gt_flat, pred_flat, types, print_results=True, save_results=False)

    gt_flat, pred_flat, types = _score(gt_ent, pred_ent, label_ind=2)
    norm_eval = _compute_metrics(gt_flat, pred_flat, types, print_results=False,save_results=True) 

    if log:
        log.info('Evaluation:')
        log.info('\n--- Entities (NER) ---')
        log.info(f'{ner_eval}')

        log.info('Evaluation:')
        log.info('\n--- Entities (NER+Norm) ---')
        log.info(f'{norm_eval}')


def _convert_by_setting(gt:List[List[Tuple]], pred:List[List[Tuple]], include_entity_types:bool=False):
    assert len(gt) == len(pred)

    # either include or remove entity types based on setting
    def convert(t):
        if not include_entity_types:
            # remove entity type and score for evaluation
            if type(t[0]) == str:  # entity
                c = [t[0], t[1], t[3]]
            else:  # relation
                c = [(t[0][0], t[0][1], 'pseudo_entity_type'),
                     (t[1][0], t[1][1], 'pseudo_entity_type'), t[2]]
        else:
            c = list(t[:3])
        return tuple(c)

    converted_gt, converted_pred = [], []
    for sample_gt, sample_pred in zip(gt, pred):
        converted_gt.append([convert(t) for t in sample_gt])
        converted_pred.append([convert(t) for t in sample_pred])

    return converted_gt, converted_pred


def _score(gt: List[List[Tuple]], pred: List[List[Tuple]], label_ind:int=2):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    types = set()


    for (sample_gt, sample_pred) in zip(gt, pred):

        union = set()
        union.update(sample_gt)
        union.update(sample_pred)

        for s in union:
#            print(s)
            if s in sample_gt:
                t = s[label_ind]
                gt_flat.append(t)
                types.add(t)
            else:
                gt_flat.append(0)

            if s in sample_pred:
                t = s[label_ind]
                pred_flat.append(t)
                types.add(t)
            else:
                pred_flat.append(0)

    return gt_flat, pred_flat, types

def _compute_metrics(gt_all, pred_all, types, print_results:bool=False, save_results:bool=False):
    labels = list(types)
    per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
    total_support = sum(per_type[-1])

    res_str=0
    if print_results:
        res_str = _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

    if save_results:
        dct={}
        dct['type']=labels+['micro','macro']
        dct['precision']=list(per_type[0])+[micro[0],macro[0]]
        dct['recall']=list(per_type[1])+[micro[1],macro[1]]
        dct['fscore']=list(per_type[2])+[micro[2],macro[2]]
        dct['support'] = list(per_type[3]) + [total_support,total_support]
        dct=pd.DataFrame.from_dict(dct)
        dct.to_csv(args.output_path+'prfs.csv',index=False)

    return res_str


def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)

    return results_str

def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("-t", dest="true", default="./true.json", type=str,
                             help="Path to the gold-true labels in JSON format file (default: 'true.json')")
    args_parser.add_argument("-p", dest="pred", default="./pred.json", type=str,
                             help="Path to the predicted labels in JSON format file (default: 'pred.json')")
    args_parser.add_argument("-o", dest="output_path", default="./", type=str,
                             help="Path to store output log (default: './')")
    args_parser.add_argument('--i', action='store_false', dest="include_concept_less",
                             help="Bool-type flag for either consider entity tags in accuracy evaluation(needed for ADR/Indication f1 scores calculation). Set to False if you don't want such entities to be considered.")
    args_parser.add_argument('--c', action='store_true', dest="consider_entity_tags",
                             help="Bool-type flag for either consider entity tags in accuracy evaluation(needed for ADR/Indication f1 scores calculation). Set to False if you don't want such entities to be considered.")
    args = args_parser.parse_args()

    logging.basicConfig(format=u'%(message)s', filemode='w', level=logging.INFO,
                        filename=args.output_path + "output_log.txt")

    print(f"Parse data from {args.true}")

    gt_ent = read_json(args.true, include_concept_less=args.include_concept_less,consider_entity_tags=args.consider_entity_tags)

    print(f"Parse data from {args.pred}\n")
    
    pred_ent = read_json(args.pred,include_concept_less=args.include_concept_less,consider_entity_tags=args.consider_entity_tags)

    compute_scores(gt_ent, pred_ent, log=logging,consider_entity_tags=args.consider_entity_tags)

    logging.info('\nSuccessful end.')