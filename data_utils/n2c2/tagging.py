import os
from sys import path
path.append(os.getcwd())
from data_utils.n2c2.preprocessing import replace_ponctuation_with_space, process, split_on_uppercase, tokenize_sentence
from nltk.tokenize import word_tokenize

LABELSET = {'Drug': {'b': 'B-Drug', 'i': 'I-Drug', 'l': 'L-Drug', 'u': 'U-Drug'},
             'Strength': {'b': 'B-Strength', 'i': 'I-Strength', 'l': 'L-Strength', 'u': 'U-Strength'},
            'Dosage': {'b': 'B-Dosage', 'i': 'I-Dosage', 'l': 'L-Dosage', 'u': 'U-Dosage'},
            'Duration': {'b': 'B-Duration', 'i': 'I-Duration', 'l': 'L-Duration', 'u': 'U-Duration'},
            'Frequency': {'b': 'B-Frequency', 'i': 'I-Frequency', 'l': 'L-Frequency', 'u': 'U-Frequency'},
            'Form': {'b': 'B-Form', 'i': 'I-Form', 'l': 'L-Form', 'u': 'U-Form'},
            'Route': {'b': 'B-Route', 'i': 'I-Route', 'l': 'L-Route', 'u': 'U-Route'},
             'Reason': {'b': 'B-Reason', 'i': 'I-Reason', 'l': 'L-Reason', 'u': 'U-Reason'},
             'ADE': {'b': 'B-ADE', 'i': 'I-ADE', 'l': 'L-ADE', 'u': 'U-ADE'}}
			 
			 
def tag_sentence(tok_text, tok_entity, begin, labelset, labels, entity_start, relation_type, segment):
    if segment == "BIO":
        for i, (word, start) in enumerate(zip(tok_text, entity_start)):
            if start == begin: 
                if len(tok_entity) == 1:
                    labels[i] = labelset['b'] + relation_type
                else:
                    labels[i] = labelset['b'] + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i'] + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['i'] + relation_type
    elif segment == "BILOU":
        for i, (word, start) in enumerate(zip(tok_text, entity_start)):
            if start == begin: 
                if len(tok_entity) == 1:
                    labels[i] = labelset['u'] + relation_type
                else:
                    labels[i] = labelset['b'] + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i'] + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['l'] + relation_type
    return labels
			
def return_label(lab, string, labelset, tok_text, labels, mstart, entity_start, start, end, ss, relation_type, segment):
    string = process(replace_ponctuation_with_space(string))
    tok_mention = tokenize_sentence(string)
                        
    mention_str_len = len(string.lstrip())
    index = len(tok_mention[0]) - mention_str_len

    if mstart in entity_start:                            
        labels = tag_sentence(tok_text, tok_mention, mstart, labelset, labels, entity_start, relation_type, segment)
    elif (mstart + index) in entity_start:
        labels = tag_sentence(tok_text, tok_mention, mstart + index, labelset, labels, entity_start, relation_type, segment)
    elif (mstart + index) in range(start, end):
        labels = tag_sentence(tok_text, tok_mention, ss.find(tok_mention[0]) + start, labelset, labels, entity_start, relation_type, segment)
    return labels

def tagging_sequence_1(lab, set_ADE_mention, sequence_1, tok_text, entity_start, start, end, sentence, relation_type, segment):
    for m in set_ADE_mention:
        mstart = m.start                   
        mend = m.end    
        m_str = m.text

        sequence_1 = return_label(lab, m_str, LABELSET[m.ttype], tok_text, sequence_1, mstart, entity_start, start, end, sentence, relation_type, segment)                                
    return sequence_1

def replace_tokens_with_ADE_type(lab, drug, tok_text, entity_start, start, end, sentence, segment):
    mstart = drug.start                  
    mend = drug.end 
    m_str = drug.text
    sentence_2 = tokenize_sentence(sentence) 
    sentence_2 = return_label(lab, m_str, LABELSET[drug.ttype], tok_text, sentence_2, mstart, entity_start, start, end, sentence, '', segment)                                
    return sentence_2

def tagging_sequence_2(lab, drug, attributes, sentence, start, tok_text, end, entity_start, segment):
    #Replace ade string with type in the sentence
    sentence_2 = replace_tokens_with_ADE_type(lab, drug, tok_text, entity_start, start, end, sentence, segment)
    #Set sequence 2
    
    sequence_2 = tagging_sequence_1(lab, attributes, ['O'] * len(tok_text), tok_text, entity_start, start, end, sentence, '', segment)
    return sentence_2, sequence_2

			
			
			
			
