from collections import defaultdict
from datetime import datetime
import pickle
import re

import pandas as pd

import spacy
NLP = spacy.load('en_core_web_sm')  
LEMMATIZER = NLP.get_pipe("lemmatizer")


def get_lemma_spacy(verb):
    verb = verb.split()[0]
    _lemmas = LEMMATIZER.lemmatize(NLP(verb)[0])
    return _lemmas[0]


def get_verb_agency_dict(agency_path):

    verb_agency_dict = {}

    agency_df = pd.read_csv(agency_path)

    for _index, _row in agency_df.iterrows():
        _verb = _row['verb']
        _lemma = get_lemma_spacy(_verb)
        verb_agency_dict[_lemma] = _row['agency']

    return verb_agency_dict


def get_verb_power_dict(agency_path):

    verb_power_dict = {}

    agency_df = pd.read_csv(agency_path)

    for _index, _row in agency_df.iterrows():
        _verb = _row['verb']
        _lemma = get_lemma_spacy(_verb)
        verb_power_dict[_lemma] = _row['power']

    return verb_power_dict


def get_persona_matches_per_document(parsed_doc, persona_pattern_dict):

    nsubj_verb_count_dict = defaultdict(int)
    dobj_verb_count_dict = defaultdict(int)

    for _parsed_sentence in parsed_doc.sents:
        for _noun_chunk in _parsed_sentence.noun_chunks:   

            if _noun_chunk.root.dep_ == 'nsubj':

                for _persona, _pattern in persona_pattern_dict.items():

                    if re.findall(_pattern, _noun_chunk.text.lower()):

                        _nusbj = _persona
                        _verb = _noun_chunk.root.head.lemma_.lower()
                        nsubj_verb_count_dict[(_nusbj, _verb)] += 1     

            elif _noun_chunk.root.dep_ == 'dobj':

                for _persona, _pattern in persona_pattern_dict.items():

                    if re.findall(_pattern, _noun_chunk.text.lower()):

                        _dobj = _persona
                        _verb = _noun_chunk.root.head.lemma_.lower() 
                        dobj_verb_count_dict[(_dobj, _verb)] += 1   

    return nsubj_verb_count_dict, dobj_verb_count_dict


def get_persona_totals_per_document(nsubj_verb_count_dict, 
                                     dobj_verb_count_dict):

    persona_total_dict = defaultdict(int)
    
    for (_persona, _verb), _count in nsubj_verb_count_dict.items():
        persona_total_dict[_persona] += _count
    for (_persona, _verb), _count in dobj_verb_count_dict.items():
        persona_total_dict[_persona] += _count

    return persona_total_dict


def measure_power_per_document(nsubj_verb_count_dict, 
                               dobj_verb_count_dict, 
                               verb_power_dict):

    persona_power_dict = defaultdict(lambda: defaultdict(int))

    for (_persona, _verb), _count in nsubj_verb_count_dict.items():
        if _verb in verb_power_dict:
            _power = verb_power_dict[_verb]  
            if _power == 'power_agent':
                persona_power_dict[_persona]['positive'] += _count
            if _power == 'power_theme':
                persona_power_dict[_persona]['negative'] += _count

    for (_persona, _verb), _count in dobj_verb_count_dict.items():
        if _verb in verb_power_dict:
            _power = verb_power_dict[_verb]  
            if _power == 'power_theme':
                persona_power_dict[_persona]['positive'] += _count
            if _power == 'power_agent':
                persona_power_dict[_persona]['negative'] += _count

    return persona_power_dict


def measure_power(verb_power_dict, persona_pattern_dict, texts, text_ids):

    id_nsubj_verb_count_dict = {}
    id_dobj_verb_count_dict = {}
    id_persona_power_dict = {}
    id_persona_total_dict = {}

    j = 0

    for _text, _id in zip(texts, text_ids):

        if j % 100 == 0:
            print(str(datetime.now())[:-7] + ' Processed ' + str(j) + ' out of ' + str(len(texts)))
        j += 1

        _parse = NLP(_text)
        
        _nsubj_verb_count_dict, _dobj_verb_count_dict = get_persona_matches_per_document(_parse, persona_pattern_dict)
        _persona_power_dict = measure_power_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict, verb_power_dict)
        _persona_total_dict = get_persona_totals_per_document(_nsubj_verb_count_dict, _dobj_verb_count_dict)

        id_persona_power_dict[_id] = _persona_power_dict
        id_persona_total_dict[_id] = _persona_total_dict
        id_nsubj_verb_count_dict[_id] = _nsubj_verb_count_dict
        id_dobj_verb_count_dict[_id] = _dobj_verb_count_dict

    return id_persona_power_dict, id_persona_total_dict, id_nsubj_verb_count_dict, id_dobj_verb_count_dict


def evaluate_persona_coverage(id_persona_total_dict, id_persona_power_dict):

    persona_found_dict = defaultdict(int)
    persona_missed_dict = defaultdict(int)
    persona_total_dict = defaultdict(int)

    for _id, _persona_power_dict in id_persona_power_dict.items():

        for _persona in _persona_power_dict.keys():

            _total = id_persona_total_dict[_id][_persona]
            _found = _persona_power_dict[_persona]['positive'] + _persona_power_dict[_persona]['negative']

            _missed = _total - _found

            persona_found_dict[_persona] += _found
            persona_missed_dict[_persona] += _missed
            persona_total_dict[_persona] += _total

    return persona_found_dict, persona_missed_dict, persona_total_dict


def evaluate_verb_coverage(id_nsubj_verb_count_dict):

    verb_count_dict = defaultdict(int)

    for _id, _nsubj_verb_count_dict in id_nsubj_verb_count_dict.items():

        for (_persona, _verb), _count in _nsubj_verb_count_dict.items():

            verb_count_dict[_verb] += 1
    
    return verb_count_dict