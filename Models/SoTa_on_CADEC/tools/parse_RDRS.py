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
    '''Возвращает предложения и их спаны'''
    text_rows = message.split('\n')
    assert type(sent_tokenizer) == nltk.tokenize.PunktSentenceTokenizer, 'В качестве sent_tokenizer поддерживается только PunktSentenceTokenizer'
    #к сожалению PunktSentenceTokenizer не токенизирует по \n, приходится выкручиваться
    text_rows_tokenized = [[list(sent_span) for sent_span in sent_tokenizer.span_tokenize(text_row)] for text_row in text_rows]
    empty_rows = 0
    for i, tokenized_row in enumerate(text_rows_tokenized):
        if i==0:
            continue
        #восстанавливаем правильные спаны
        if tokenized_row==[]:
            #в таком случае два \n шли подряд
            tokenized_row.append(['not a sent', text_rows_tokenized[i-1][-1][-1]+1])
            continue
        for sent in text_rows_tokenized[i]:
            sent[0]+=text_rows_tokenized[i-1][-1][-1]+1 #индекс самой последней буквы последнего предложения + 1 на знак \n
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
    log_markup_errors = [] #сущности, которые не нашлись в MedDRA
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
                    #ищем в каком предложении и находим спаны сущности в этом предложении
                    ent_begin = ent['spans'][0]['begin']
                    ent_end = ent['spans'][-1]['end']
                    for sent_id, sentence_span in enumerate(doc_sentences_spans):
                        if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                            break
                    else:
                        #предложение с сущностью не найдено, не удалось установить его границы
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
                #первым делом ищем предложение с сущностью
                if add_context:
                    #ищем в каком предложении и находим спаны сущности в этом предложении
                    ent_begin = ent['spans'][0]['begin']
                    ent_end = ent['spans'][-1]['end']
                    for sent_id, sentence_span in enumerate(doc_sentences_spans):
                        if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                            break
                    else:
                        #предложение с сущностью не найдено, не удалось установить его границы
                        log_markup_errors.append({'review_id': review['meta']['fileName'], 
                                                  'entity_id': ent['xmiID'],
                                                  'entity_text': ent['text'],
                                                  'entity_concept_term': ent['MedDRA'],
                                                  'set': 'test',
                                                  'error': 'no sent borders found'})
                        continue
                    phrase_spans_in_sentence = [[span['begin'] - sentence_span[0], 
                                                span['end'] - sentence_span[0]] for span in ent['spans']]
                #ветка с тестом с conceptless
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
                    #уже произошла обработка предыдущей веткой, продолжаем
                    ent['MedDRA'] = ''
                    continue
                #наконец обычная ветка с тестом, такая же как в трейне
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
                    #первым делом ищем предложение
                    if add_context:
                        #ищем в каком предложении и находим спаны сущности в этом предложении
                        ent_begin = ent['spans'][0]['begin']
                        ent_end = ent['spans'][-1]['end']
                        for sent_id, sentence_span in enumerate(doc_sentences_spans):
                            if ent_begin >= sentence_span[0] and ent_end <= sentence_span[1]:
                                break
                        else:
                            #предложение с сущностью не найдено, не удалось установить его границы
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
