import os
import spacy
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree
from sklearn.model_selection import ShuffleSplit
from sys import path
path.append(os.getcwd())
from data_utils.n2c2.ehr import Corpora
from data_utils.n2c2.utils import get_concepts, get_relations_from_sentence
from data_utils.n2c2.preprocessing import replace_ponctuation_with_space, process, split_paragraph, tokenize_sentence, spans
from data_utils.n2c2.tagging import tagging_sequence_2

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = nlp.tokenizer.tokens_from_list

def read_n2c2_corpus(dir_name_notes, segment="BIO"):
    att_relation_data = []
    l = 0
    files = Corpora(dir_name_notes, 2)    
    for i in range(len(files.docs)):
        text = files.docs[i]._get_text()
        concepts = files.docs[i].annotations['tags']
        relations = files.docs[i].annotations['relations']
        
        sentence_file = files.docs[i].basename
        sentence_file = sentence_file.replace('.ann', '.xml')
        #print (sentence_file)
        sentences = split_paragraph(text)
        l = l + len(sentences)
        for s in sentences:
            start_sent = text.index(s)
            end_sent = start_sent + len(s)
            attributes, drugs, concept_sent = get_concepts(concepts, start_sent, end_sent)
            
            ade_relation = get_relations_from_sentence(drugs, attributes, relations)
                
            if len(ade_relation)>0:
                for d, att in ade_relation:
                    concept_sent = []
                    concept_sent.extend(att)
                    concept_sent.append(d)
                    
                    a_start = min([a.start for a in concept_sent])
                    a_end = max([a.end for a in concept_sent])

                    st = text[a_start:a_end]
                    start_sent = a_start


                    sent = process(replace_ponctuation_with_space(st))
                    tok_text = tokenize_sentence(sent)

                    entity_start, entity_end = spans(sent, tok_text, start_sent)

                    entity_file = [sentence_file.replace('.xml', '')]*len(tok_text)
                
                    d1 = (str(d.end), str(d.start), "Drug", d.text)
                    att1 = []
                    for a in att:
                        att1.append((str(a.end), str(a.start), a.ttype, a.text))
                    sentence_2, sequence_2 = tagging_sequence_2(sentence_file, 
                                                                d, 
                                                                att, 
                                                                sent, start_sent, tok_text, end_sent, entity_start, segment)
                    att_relation_data.append((sentence_2,tok_text, sequence_2,entity_start,entity_file,entity_end,d1, att1, entity_file))


    
    return att_relation_data

def split_n2c2_data(sentences_train):
    N2C2_dev = N2C2()
    N2C2_train_temp = N2C2()
    
    rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

    for train_index, test_index in rs.split(sentences_train.t_sentence_relation):
        N2C2_dev.t_sentence_relation = [sentences_train.t_sentence_relation[i] for i in test_index]
        N2C2_dev.t_segment_relation = [sentences_train.t_segment_relation[i] for i in test_index]
        N2C2_dev.t_toks_relation = [sentences_train.t_toks_relation[i] for i in test_index]
        N2C2_dev.t_start_relation = [sentences_train.t_start_relation[i] for i in test_index]
        N2C2_dev.t_section_relation = [sentences_train.t_section_relation[i] for i in test_index]
        N2C2_dev.t_len_relation = [sentences_train.t_len_relation[i] for i in test_index]
        N2C2_dev.t_ade = [sentences_train.t_ade[i] for i in test_index]
        N2C2_dev.t_modifiers = [sentences_train.t_modifiers[i] for i in test_index]
        N2C2_dev.t_drug_relation = [sentences_train.t_drug_relation[i] for i in test_index]
        N2C2_dev.heads = [sentences_train.heads[i] for i in test_index]
        N2C2_dev.deps = [sentences_train.deps[i] for i in test_index]
		
        N2C2_train_temp.t_sentence_relation = [sentences_train.t_sentence_relation[i] for i in train_index]
        N2C2_train_temp.t_segment_relation = [sentences_train.t_segment_relation[i] for i in train_index]
        N2C2_train_temp.t_toks_relation = [sentences_train.t_toks_relation[i] for i in train_index]
        N2C2_train_temp.t_start_relation = [sentences_train.t_start_relation[i] for i in train_index]
        N2C2_train_temp.t_section_relation = [sentences_train.t_section_relation[i] for i in train_index]
        N2C2_train_temp.t_len_relation = [sentences_train.t_len_relation[i] for i in train_index]
        N2C2_train_temp.t_ade = [sentences_train.t_ade[i] for i in train_index]
        N2C2_train_temp.t_modifiers = [sentences_train.t_modifiers[i] for i in train_index]
        N2C2_train_temp.t_drug_relation = [sentences_train.t_drug_relation[i] for i in train_index]
        N2C2_train_temp.heads = [sentences_train.heads[i] for i in train_index]
        N2C2_train_temp.deps = [sentences_train.deps[i] for i in train_index]


    return N2C2_train_temp, N2C2_dev 

def append_data(data_y, max_level):
    data_y_final = []
    for labels in data_y:
        y= []
        y.extend(labels)
        if len(y)!=max_level:
            for i in range(len(y),max_level):
                y.append(labels[0])
        data_y_final.append(y)
    return data_y_final
	
class N2C2:
    def __init__(self, segment = "BIO", label_dir=None):
        self.label_dir = label_dir
        self.segment = segment

        
        self.ade_relation_data = []
        self.t_drug_relation = []
        self.t_toks_relation = []
        self.t_sentence_relation = []
        self.t_segment_relation = []
        self.t_section_relation = []
        self.t_start_relation = []
        self.t_len_relation = []
        self.t_ade = []
        self.t_modifiers = []  
        self.heads = []
        self.deps = [] 

    def load_corpus(self):

        ADE_relation_data = read_n2c2_corpus(self.label_dir, self.segment)
        self.ade_relation_data = ADE_relation_data        
        for seq in self.ade_relation_data:
            self.t_sentence_relation.append(seq[0])
            self.t_toks_relation.append(seq[1])
            self.t_segment_relation.append([seq[2]])
            self.t_start_relation.append(seq[3])
            self.t_section_relation.append(seq[4])
            self.t_len_relation.append(seq[5])
            self.t_ade.append(seq[6])
            self.t_modifiers.append(seq[7])
            self.t_drug_relation.append(seq[8])
                
        self.t_segment_relation = append_data(self.t_segment_relation, 3)
        for doc in nlp.pipe(self.t_sentence_relation):
            head = [token.head.i for token in doc]
            dep = [token.dep_ for token in doc]
            self.heads.append(head)
            self.deps.append(dep)         