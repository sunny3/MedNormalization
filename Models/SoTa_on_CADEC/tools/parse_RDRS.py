import jsonlines
import numpy as np
from sklearn.model_selection import train_test_split
import nltk

def parse_RDRS(RDRS_path):
    ds = []
    with jsonlines.open(RDRS_path) as reader:
        for obj in reader:
            ds.append(obj)
    return ds

def split_sentences(message, sent_tokenizer):
    '''Return sentences and their spans'''
    text_rows = message.split('\n')
    assert type(sent_tokenizer) == nltk.tokenize.PunktSentenceTokenizer, 'Only PunktSentenceTokenizer supported as sent_tokenizer'
    #unfortunately PunktSentenceTokenizer does not tokenize by \n, you have to get out
    text_rows_tokenized = [[list(sent_span) for sent_span in sent_tokenizer.span_tokenize(text_row)] for text_row in text_rows]
    empty_rows = 0
    for i, tokenized_row in enumerate(text_rows_tokenized):
        if i==0:
            continue
        #restore correct spans
        if tokenized_row==[]:
            #in case 2 \n in a row
            tokenized_row.append(['not a sent', text_rows_tokenized[i-1][-1][-1]+1])
            continue
        for sent in text_rows_tokenized[i]:
            sent[0]+=text_rows_tokenized[i-1][-1][-1]+1 #index of the last letter of the last sentence + 1 (for \n)
            sent[1]+=text_rows_tokenized[i-1][-1][-1]+1
    doc_sentences_spans = list(filter(lambda x: True if x[0]!='not a sent' else False, sum(text_rows_tokenized, [])))
    doc_sentences = []
    for text_row in text_rows:
        for sent_text in sent_tokenizer.sentences_from_text(text_row):
            if sent_text==[]:
                continue
            doc_sentences.append(sent_text)
    return doc_sentences_spans, doc_sentences

def parse_json_ds(X_train, X_test, CV, X_val=None, CV_with_conceptless=None, add_context=False):
    log_markup_errors = [] #entities not found in MedDRA
    train_concepts, train_phrases, train_ids = [], [], []
    test_concepts, test_phrases, test_ids = [], [], []
    if CV_with_conceptless is not None:
        test_concepts_with_conceptless, test_phrases_with_conceptless, test_ids_with_conceptless = [], [], []
        if add_context:
            test_sentences_with_conceptless = []
    if X_val is not None:
        val_concepts, val_phrases, val_ids = [], [], []
        if add_context:
            val_sentences = []
    if add_context:
        sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        train_sentences = []
        test_sentences = []
    for review in X_train:
        if add_context:
            doc_sentences_spans, doc_sentences = split_sentences(review['raw'], sent_tokenizer)
        for ent in review['objects']['MedEntity']:
            if 'MedDRA' in ent.keys():
                if ent['MedDRA']=='':
                    continue
                if add_context:
                    #look phrase's sentence and find the spans of the entity in this sentence
                    ent_begin = ent['spans'][0]['begin']
                    ent_end = ent['spans'][-1]['end']
                    for sent_id, sentence_span in enumerate(doc_sentences_spans):
                        if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                            break
                    else:
                        #sentence with entity not found, unable to set its bounds
                        log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                                  'entity_id': ent['xmiID'],
                                                  'entity_text': ent['text'],
                                                  'entity_concept_term': ent['MedDRA'],
                                                  'set': 'train',
                                                  'error': 'no sent borders found'})
                        continue
                    phrase_spans_in_sentence = [[span['begin'] - sentence_span[0], 
                                                span['end'] - sentence_span[0]] for span in ent['spans']]
                try:
                    train_concepts.append(CV.meddra_term_to_meddra_code[ent['MedDRA'].split('|')[0]])
                except KeyError:
                    log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                              'entity_id': ent['xmiID'],
                                              'entity_text': ent['text'],
                                              'entity_concept_term': ent['MedDRA'],
                                              'error': 'no match in CV meddra_terms'})
                    continue
                if add_context:
                    train_sentences.append({'sentence': doc_sentences[sent_id], 
                                            'phrase': ent['text'],
                                            'phrase_spans': phrase_spans_in_sentence})
                
                train_phrases.append(ent['text'])
                train_ids.append(review['meta']['fileName'])
    for review in X_test:
        if add_context:
            doc_sentences_spans, doc_sentences = split_sentences(review['raw'], sent_tokenizer)
        for ent in review['objects']['MedEntity']:
            if 'MedDRA' in ent.keys():
                #first of all, we are looking for a sentence with this entity
                if add_context:
                    #look phrase's sentence and find the spans of the entity in this sentence
                    ent_begin = ent['spans'][0]['begin']
                    ent_end = ent['spans'][-1]['end']
                    for sent_id, sentence_span in enumerate(doc_sentences_spans):
                        if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                            break
                    else:
                        #in case sentence with entity was not found
                        log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                                  'entity_id': ent['xmiID'],
                                                  'entity_text': ent['text'],
                                                  'entity_concept_term': ent['MedDRA'],
                                                  'set': 'test',
                                                  'error': 'no sent borders found'})
                        continue
                    phrase_spans_in_sentence = [[span['begin'] - sentence_span[0], 
                                                span['end'] - sentence_span[0]] for span in ent['spans']]
                #conceptless branch
                if ent['MedDRA']=='':
                    ent['MedDRA'] = 'CONCEPT_LESS'
                if CV_with_conceptless is not None:
                    try:
                        test_concepts_with_conceptless.append(CV_with_conceptless.meddra_term_to_meddra_code[ent['MedDRA'].split('|')[0]])
                        test_phrases_with_conceptless.append(ent['text'])
                        test_ids_with_conceptless.append(review['meta']['fileName'])
                        if add_context:
                            test_sentences_with_conceptless.append({'sentence': doc_sentences[sent_id], 
                                                                       'phrase': ent['text'],
                                                                       'phrase_spans': phrase_spans_in_sentence})
                    except KeyError:
                        log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                                  'entity_id': ent['xmiID'],
                                                  'entity_text': ent['text'],
                                                  'entity_concept_term': ent['MedDRA'],
                                                  'error': 'no match in CV_with_concepless meddra terms'})
                if ent['MedDRA']=='CONCEPT_LESS':
                    ent['MedDRA'] = ''
                    continue
                #finally, a regular branch with a test, the same as in the train
                try:
                    test_concepts.append(CV.meddra_term_to_meddra_code[ent['MedDRA'].split('|')[0]])
                except KeyError:
                    log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                              'entity_id': ent['xmiID'],
                                              'entity_text': ent['text'],
                                              'entity_concept_term': ent['MedDRA'],
                                              'error': 'no match in CV meddra terms'})
                    continue
                if add_context:
                    test_sentences.append({'sentence': doc_sentences[sent_id], 
                                            'phrase': ent['text'],
                                            'phrase_spans': phrase_spans_in_sentence})
                test_phrases.append(ent['text'])
                test_ids.append(review['meta']['fileName'])
    if X_val is not None:
        for review in X_val:
            if add_context:
                doc_sentences_spans, doc_sentences = split_sentences(review['raw'], sent_tokenizer)
            for ent in review['objects']['MedEntity']:
                if 'MedDRA' in ent.keys():
                    if ent['MedDRA']=='':
                        continue
                    #first of all, we are looking for a sentence with this entity
                    if add_context:
                        #look phrase's sentence and find the spans of the entity in this sentence
                        ent_begin = ent['spans'][0]['begin']
                        ent_end = ent['spans'][-1]['end']
                        for sent_id, sentence_span in enumerate(doc_sentences_spans):
                            if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                                break
                        else:
                            #in case sentence with entity was not found
                            log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                                      'entity_id': ent['xmiID'],
                                                      'entity_text': ent['text'],
                                                      'entity_concept_term': ent['MedDRA'],
                                                      'set': 'valid',
                                                      'error': 'no sent borders found'})
                            continue
                        phrase_spans_in_sentence = [[span['begin'] - sentence_span[0], 
                                                span['end'] - sentence_span[0]] for span in ent['spans']]
                    try:
                        val_concepts.append(CV.meddra_term_to_meddra_code[ent['MedDRA'].split('|')[0]])
                    except KeyError:
                        log_markup_errors.append({'review_id':  review['meta']['fileName'], 
                                                  'entity_id': ent['xmiID'],
                                                  'entity_text': ent['text'],
                                                  'entity_concept_term': ent['MedDRA'],
                                                  'error': 'no match in CV meddra terms'})
                        continue
                    if add_context:
                        val_sentences.append({'sentence': doc_sentences[sent_id], 
                                               'phrase': ent['text'],
                                               'phrase_spans': phrase_spans_in_sentence})
                    val_phrases.append(ent['text'])
                    val_ids.append(review['meta']['fileName'])
                        
    ds = {'train': {'phrases': train_phrases, 'concepts': train_concepts, 'train_ids': train_ids}, 
          'test': {'phrases': test_phrases, 'concepts': test_concepts, 'train_ids': test_ids}}
    if add_context:
        ds['train']['sentences'] = train_sentences
        ds['test']['sentences'] = test_sentences
    if X_val is not None:
        ds['validation'] = {'phrases': val_phrases, 'concepts': val_concepts, 'train_ids': val_ids}
        if add_context:
            ds['validation']['sentences'] = val_sentences
    if CV_with_conceptless is not None:
        ds['test_with_conceptless'] = {'phrases': test_phrases_with_conceptless, 
                                       'concepts': test_concepts_with_conceptless, 
                                       'train_ids': test_ids_with_conceptless}
        if add_context:
            ds['test_with_conceptless']['sentences'] = test_sentences_with_conceptless
    return ds, log_markup_errors
    
def CADEC_format_to_RDRS_format(CV, cadec_path='../../Data/Raw/CADEC_origin/'):
    '''Function converting CADEC format to RDRS format (Returns list of reviews)'''
    cadec_text_path = os.path.join(cadec_path, 'text') + '/'
    cadec_meddra_path = os.path.join(cadec_path, 'meddra') + '/'
    all_reviews = []
    for ann_file in os.listdir(cadec_meddra_path):
        ann_filename, ann_file_extension = os.path.splitext(cadec_meddra_path + ann_file)
        if ann_file_extension != '.ann':
            continue
        new_rev = {'meta': {'fileName': ann_file}, 
                   'raw': '', 
                   'objects': {'MedEntity': []}}
        text_file_path = cadec_text_path + ann_file[:-4] + '.txt'
        ann_file_path = cadec_meddra_path + ann_file
        with open(text_file_path) as f:
            review_text = f.read()
        new_rev['raw'] = review_text
        i=0
        with open(ann_file_path) as f:
            for line in f:
                if line=='':
                    continue
                line = re.sub('\s\s+', '\t', line)
                ent_id, markup_pt_inf, ent_text = line.split('\t')
                markup_pt_inf = re.sub('\s\+\s|/', '+', markup_pt_inf)
                
                try:
                    pt_id = '|'.join(map(lambda x: CV.meddra_code_to_meddra_term[x], \
                                     markup_pt_inf.split(' ')[0].split('+')))
                except KeyError:
                    continue
                spans = []
                markup_pt_inf = re.sub('^\+*(\d{8,9}|CONCEPT_LESS)(\+(\d{8,9}|CONCEPT_LESS))*\s|^CONCEPT_LESS\s', '', markup_pt_inf)
                if markup_pt_inf.find(';')>=0:
                    for ent_span in markup_pt_inf.split(';'):
                        ent_begin, ent_end =  ent_span.split(' ')
                        ent_begin, ent_end = int(ent_begin), int(ent_end) 
                        spans.append({'begin': ent_begin, 
                                      'end': ent_end })
                else:
                    ent_begin, ent_end = markup_pt_inf.split(' ')
                    spans.append({
                        'begin': int(ent_begin), 
                        'end': int(ent_end)
                    })
                new_rev['objects']['MedEntity'].append({
                    'xmiID': ent_id,
                    'spans': spans,
                    'text': ent_text.strip('\n'),
                    'MedDRA': pt_id if pt_id!='CONCEPT_LESS' else ''
                })
        all_reviews.append(new_rev)
    return all_reviews
    
def sort_concepts_func(curr_ds):
        '''
        Concepts in the markup, if there are multiple mentions, are separated by '|'
        In the model, we take one concept, the first one in the list
        Different mentions can share a common concept, but due to the '|' separation,
        the model may not recognize this shared concept, which is a labeling omission
        The function corrects the markup so that shared concepts always come first
        '''

        all_concepts = []

        for rev in curr_ds:
            for ent in rev['objects']['MedEntity']:
                if 'MedDRA' in ent.keys() and ent['MedDRA']!='':
                    all_concepts.extend(ent['MedDRA'].split('|'))
        
        concepts_popularity = Counter(all_concepts)
        concepts_popularity = {k:v for k,v in sorted(concepts_popularity.items(), reverse=True, key=lambda x: x[1])}
        
        already_processed_mentions = set()
        for concept in tqdm(concepts_popularity):
            for i, rev in enumerate(curr_ds):
                for j, ent in enumerate(rev['objects']['MedEntity']):
                    if str(i)+'_'+str(j) in already_processed_mentions:
                        continue
                    if 'MedDRA' in ent.keys() and ent['MedDRA'].find('|')>=0:
                        all_phrase_concepts = ent['MedDRA'].split('|')
                        if concept in all_phrase_concepts:
                            if all_phrase_concepts.index(concept)!=0:
                                all_phrase_concepts.remove(concept)
                                all_phrase_concepts.insert(0,concept)
                                ent['MedDRA'] = '|'.join(all_phrase_concepts)
                            already_processed_mentions.add(str(i)+'_'+str(j))
                        
        return curr_ds
