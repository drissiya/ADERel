import os
from sys import path
path.append(os.getcwd())
from nltk.tokenize import word_tokenize
from data_utils.n2c2.ehr import Corpora

def extract_mention_from_sentence(drug, set_toks, ys_bio, section, start, leng, segment):
    data_drug = []
    data_toks = []
    data_ys = []
    data_sec = []
    data_start = []
    data_len = []
    if segment == "BIO":
        for d, tok, ys, sec, st, le in zip(drug, set_toks, ys_bio, section, start, leng):
            temp_toks = []
            temp_ys = []
            temp_sec = []
            temp_start = []
            temp_len = []
            temp_drug = []
            for i, (t, yb, s, a, b, e) in enumerate(zip(tok, ys, sec, st, le, d)):

                if yb.startswith('B-'):
                    tok_txt = t
                    ys_txt = yb[2:]
                    sec_txt = s
                    start_txt = a
                    len_text = b
                    drug_text = e
                    if (i+1) == len(ys):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                        break
                    elif ys[i+1].startswith('O') and ys[i-1].startswith('O'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    elif ys[i+1].startswith('B-') and ys[i-1].startswith('B-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    else: 
                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('I-'):
                                tok_txt += ' ' + tok[i+k+1]
                                len_text = le[i+k+1] 

                            else:
                                break
                        len_t = len_text
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_sec.append(sec_txt)
                        temp_start.append(start_txt)
                        temp_len.append(len_t)
                        temp_drug.append(drug_text)
            data_toks.append(temp_toks)
            data_ys.append(temp_ys)
            data_sec.append(temp_sec)
            data_start.append(temp_start)
            data_len.append(temp_len)
            data_drug.append(temp_drug)
    elif segment == "BILOU":
        for d, tok, ys, sec, st, le in zip(drug, set_toks, ys_bio, section, start, leng):
            temp_toks = []
            temp_ys = []
            temp_sec = []
            temp_start = []
            temp_len = []
            temp_drug = []
            for i, (t, yb, s, a, b, e) in enumerate(zip(tok, ys, sec, st, le, d)):

                if yb.startswith('U-'):
                    temp_toks.append(t)
                    temp_ys.append(yb[2:])
                    temp_sec.append(s)
                    temp_start.append(a)
                    temp_len.append(b)
                    temp_drug.append(e)
                elif yb.startswith('B-'):
                    tok_txt = t
                    ys_txt = yb[2:]
                    sec_txt = s
                    start_txt = a
                    len_text = b
                    drug_text = e
                    if (i+1) == len(ys_bio):
                        break
                    else: 
                        start_n = a
                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('I-'):
                                tok_txt += ' ' + tok[i+k+1]
                                len_text = le[i+k+1] 
                                start_n = st[i+k+1]

                            elif j.startswith('L-'):
                                tok_txt += ' ' + tok[i+k+1]
                                len_text = le[i+k+1] 
                                start_n = st[i+k+1]
                                break
                        len_t = (start_n-start_txt)+len_text
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_sec.append(sec_txt)
                        temp_start.append(start_txt)
                        temp_len.append(len_t)
                        temp_drug.append(drug_text)
            data_toks.append(temp_toks)
            data_ys.append(temp_ys)
            data_sec.append(temp_sec)
            data_start.append(temp_start)
            data_len.append(temp_len)
            data_drug.append(temp_drug)

    return data_drug, data_toks, data_ys, data_sec, data_start, data_len
	
def drug_mentions(m, n2c2_guess):
    drug_rel_mention = [] 
    id_mention = 1
    dr = dict()
    for file, ade in zip(n2c2_guess.t_drug_relation, n2c2_guess.t_ade):
        if len(set(file))==0:
            continue
        if list(set(file))[0]==m:
            d = (ade[0], ade[1], ade[2], ade[3], "T"+str(id_mention))  
            dr[(ade[0], ade[1], ade[2], ade[3])] = "T"+str(id_mention)
            drug_rel_mention.append(d)
            id_mention+=1
    return drug_rel_mention, id_mention, dr

def n2c2_extract_guess_relation(TAC_guess, segment, gold_dir):
    dict_modifiers = {}
    dict_relations = {}
    dict_ade = {}


    drug_relation, toks_relation, type_relation, sec_relation, start_relation, len_relation = extract_mention_from_sentence(TAC_guess.t_drug_relation, TAC_guess.t_toks_relation, TAC_guess.t_segment_relation, TAC_guess.t_section_relation, TAC_guess.t_start_relation, TAC_guess.t_len_relation,segment)
    
    
    files = Corpora(gold_dir, 2)  
    for i in range(len(files.docs)):       
        sentence_file = files.docs[i].basename
        key = sentence_file.replace('.ann', '')
        
        
        ADE_mention, id_mention, ad = drug_mentions(key, TAC_guess)
        relations, modifiers = extract_modifiers_relation_from_label(key, ad, id_mention + 1, TAC_guess, drug_relation, toks_relation, type_relation, sec_relation, start_relation, len_relation)
        dict_modifiers[key] = modifiers
        dict_relations[key] = relations
        dict_ade[key] = ADE_mention
    return dict_ade, dict_modifiers, dict_relations

def extract_modifiers_relation_from_label(m, dict_ade, id_mention, TAC_guess, drug_relation, toks_relation, type_relation, sec_relation, start_relation, len_relation):
    id_relation = 1
    modifiers = []
    relations = []
    m_dict = []
    for ade, drug_r, toks_r, type_r, sec_r, start_r, len_r in zip(TAC_guess.t_ade, drug_relation, toks_relation,type_relation, sec_relation, start_relation, len_relation):
        if len(set(drug_r))==0:
            continue
        if list(set(drug_r))[0]==m:
            arg1 = dict_ade[(ade[0], ade[1], ade[2], ade[3])]

            for tok, typ, st, le in zip(toks_r, type_r, start_r, len_r): 
                relation_type = typ + '-Drug'
                
                #if (str(le), str(st), modifier_type, tok, sec) not in m_dict:
                arg2 = "T"+str(id_mention) 

                id_mention = id_mention + 1
                    
                modifier = (str(le), str(st), typ, ' '.join(word_tokenize(tok)), arg2)
                
                modifiers.append(modifier)
                relations.append(("R"+str(id_relation), arg1, arg2, relation_type))

                id_relation = id_relation + 1
    #print ("==============")
    return relations, modifiers