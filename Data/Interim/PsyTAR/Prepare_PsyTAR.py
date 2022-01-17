import jsonlines
import json
import os

for fold in range(1, 6):
    os.makedirs('./crossval_balanced_by_num_ADR/' + str(fold), exist_ok=True)
    os.makedirs('./crossval_balanced_by_num_reviews/' + str(fold), exist_ok=True)
    
with open('./balanced_by_num_ADR_ids.json', 'r') as f:
    fold_ids_ADR_balanced = json.load(f)
    
with open('./balanced_by_num_reviews_ids.json', 'r') as f:
    fold_ids_reviews_balanced = json.load(f)

data = []
with jsonlines.open('../../Raw/PsyTAR/PsyTAR.jsonlines') as reader:
    for obj in reader:
        data.append(obj)

#with open('../../Raw/PsyTAR/PsyTAR.jsonlines', 'r') as f:
#    data = json.load(f)
    
for fold in range(1, 6):
    test_ADR_balanced = []
    train_ADR_balanced = []
    test_reviews_balanced = []
    train_reviews_balanced = []
    for rev in data:
        rev_id = rev['meta']['fileName']
        if rev_id in fold_ids_ADR_balanced[str(fold)]:
            test_ADR_balanced.append(rev)
        else:
            train_ADR_balanced.append(rev)
        if rev_id in fold_ids_reviews_balanced[str(fold)]:
            test_reviews_balanced.append(rev)
        else:
            train_reviews_balanced.append(rev)
    with jsonlines.open('./crossval_balanced_by_num_ADR/%s/test.jsonlines'%fold, mode='w') as writer:
        for rev in test_ADR_balanced:
            writer.write(rev)
    with jsonlines.open('./crossval_balanced_by_num_ADR/%s/train.jsonlines'%fold, mode='w') as writer:
        for rev in train_ADR_balanced:
            writer.write(rev)
    with jsonlines.open('./crossval_balanced_by_num_reviews/%s/test.jsonlines'%fold, mode='w') as writer:
        for rev in test_reviews_balanced:
            writer.write(rev)
    with jsonlines.open('./crossval_balanced_by_num_reviews/%s/train.jsonlines'%fold, mode='w') as writer:
        for rev in train_reviews_balanced:
            writer.write(rev)       
    #with open('./crossval_balanced_by_num_ADR/%s/test.json'%fold, 'w') as f:
    #    json.dump(test_ADR_balanced, f)
    #with open('./crossval_balanced_by_num_ADR/%s/train.json'%fold, 'w') as f:
    #    json.dump(train_ADR_balanced, f)
    #with open('./crossval_balanced_by_num_reviews/%s/test.json'%fold, 'w') as f:
    #    json.dump(test_reviews_balanced, f)
    #with open('./crossval_balanced_by_num_reviews/%s/train.json'%fold, 'w') as f:
    #    json.dump(train_reviews_balanced, f)

print("Saved in crossval_balanced_by_num_ADR and crossval_balanced_by_num_reviews.")