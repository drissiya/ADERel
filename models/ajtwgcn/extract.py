import os
from data_utils.tac.drug_label import xml_files

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
                    elif ys[i+1].startswith('DB-') and ys[i-1].startswith('DB-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    elif ys[i+1].startswith('DI-') and ys[i-1].startswith('DI-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    elif ys[i+1].startswith('DB-') and ys[i-1].startswith('DI-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    elif ys[i+1].startswith('DI-') and ys[i-1].startswith('DB-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_sec.append(s)
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_drug.append(e)
                    else: 
                        start_n = a
                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('I-'):
                                tok_txt += ' ' + tok[i+k+1]
                                len_text = le[i+k+1] 
                                start_n = st[i+k+1]

                            else:
                                break
                        len_t = (start_n-start_txt)+len_text
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_sec.append(sec_txt)
                        temp_start.append(start_txt)
                        temp_len.append(len_t)
                        temp_drug.append(drug_text)
                elif yb.startswith('DB-'):
                    tok_txt = t
                    ys_txt = yb[3:]
                    sec_txt = s
                    start_txt = a
                    start_t = str(a)
                    len_text = b
                    drug_text = e
                    l = i
                    len_text_list = []
                    for k,j in enumerate(ys[i+1:]):
                        if j.startswith('DI-'):
                            len_text += le[i+k+1] + 1
                            if (i+k+2) == len(ys):
                                len_text_list.append(len_text)
                        elif j.startswith('DB-'):
                            len_text_list.append(len_text)
                            break
                        else:
                            len_text_list.append(len_text)
                            len_text = 0
                    if len(len_text_list)!=0:
                        len_t = str(len_text_list[0])
                        for m in len_text_list[1:]:
                            if m!=0:
                                len_t += ',' + str(m-1)

                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('DI-'):
                                tok_txt += ' ' + tok[i+k+1]                      
                                if ys[l].startswith('B-') or ys[l].startswith('I-') or ys[l].startswith('O'):
                                    start_t += ',' + str(st[i+k+1])                       

                            elif j.startswith('DB-'):
                                break
                            l = l + 1
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_sec.append(sec_txt)
                        temp_start.append(start_t)
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
                elif yb.startswith('DB-'):
                    tok_txt = t
                    ys_txt = yb[3:]
                    sec_txt = s
                    start_txt = a
                    start_t = str(a)
                    len_text = b
                    drug_text = e
                    l = i
                    len_text_list = []
                    for k,j in enumerate(ys[i+1:]):
                        if j.startswith('DI-'):
                            len_text += le[i+k+1] + 1
                            if (i+k+2) == len(ys):
                                len_text_list.append(len_text)
                        elif j.startswith('DL-'):
                            len_text += le[i+k+1] + 1
                            len_text_list.append(len_text)
                            break
                        else:
                            len_text_list.append(len_text)
                            len_text = 0
                    if len(len_text_list)!=0:
                        len_t = str(len_text_list[0])
                        for m in len_text_list[1:]:
                            if m!=0:
                                len_t += ',' + str(m-1)

                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('DI-'):
                                tok_txt += ' ' + tok[i+k+1]                      
                                if ys[l].startswith('B-') or ys[l].startswith('I-') or ys[l].startswith('O'):
                                    start_t += ',' + str(st[i+k+1])                       

                            elif j.startswith('DL-'):
                                tok_txt += ' ' + tok[i+k+1]                      
                                if ys[l].startswith('B-') or ys[l].startswith('I-') or ys[l].startswith('O'):
                                    start_t += ',' + str(st[i+k+1]) 
                                break
                            l = l + 1
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_sec.append(sec_txt)
                        temp_start.append(start_t)
                        temp_len.append(len_t)
                        temp_drug.append(drug_text)
            data_toks.append(temp_toks)
            data_ys.append(temp_ys)
            data_sec.append(temp_sec)
            data_start.append(temp_start)
            data_len.append(temp_len)
            data_drug.append(temp_drug)

    return data_drug, data_toks, data_ys, data_sec, data_start, data_len

def reshape_data(data_relation, lenght):
    data = []
    for i, j, l in zip(data_relation[0], data_relation[1], lenght):
        if l==1:
            data.append(i)
        elif l==2:
            data.append(i+j)
    return data

def trim_data(data_relation):
    lenght = []
    for i, j in zip(data_relation[0], data_relation[1]):
        if i==j:
            lenght.append(1)
        elif i!=j:
            lenght.append(2)           
    return lenght

def tac_extract_guess_relations(labels, TAC_guess, segment, gold_dir):
    dict_modifiers = {}
    dict_relations = {}
    dict_ade = {}

    drug_relation = []
    toks_relation = []
    type_relation = []
    sec_relation = []
    start_relation = []
    len_relation = []
    ade_ment = []
    
    for l in labels:
        drug, toks, type_, sec, start, len_ = extract_mention_from_sentence(TAC_guess.t_drug_relation, TAC_guess.t_toks_relation, l, TAC_guess.t_section_relation, TAC_guess.t_start_relation, TAC_guess.t_len_relation,segment)
        drug_relation.append(drug)
        toks_relation.append(toks)
        type_relation.append(type_)
        sec_relation.append(sec)
        start_relation.append(start)
        len_relation.append(len_)
    
    lenght = trim_data(toks_relation)
    toks_relation = reshape_data(toks_relation, lenght)
    drug_relation = reshape_data(drug_relation, lenght)    
    type_relation = reshape_data(type_relation, lenght)
    sec_relation = reshape_data(sec_relation, lenght)
    start_relation = reshape_data(start_relation, lenght)
    len_relation = reshape_data(len_relation, lenght)
    
    
    guess_files = xml_files(gold_dir)
    for key, v in zip(guess_files.keys(), guess_files.values()):
        ADE_mention, id_mention, ad = ade_mentions(key, TAC_guess)
        relations, modifiers = extract_modifiers_relation_from_label(key, ad, id_mention + 1, TAC_guess, drug_relation, toks_relation, type_relation, sec_relation, start_relation, len_relation)
        dict_modifiers[key] = modifiers
        dict_relations[key] = relations
        dict_ade[key] = ADE_mention
    return dict_ade, dict_modifiers, dict_relations
	
def ade_mentions(m, TAC_guess):
    ADE_rel_mention = [] 
    id_mention = 1
    ad = dict()
    for drug_m, ade in zip(TAC_guess.t_drug_relation, TAC_guess.t_ade):
        if len(set(drug_m))==0:
            continue
        if list(set(drug_m))[0]==m:
            a = (ade[0], ade[1], ade[2], ade[3], ade[4], "M"+str(id_mention))  
            ad[(ade[0], ade[1], ade[2], ade[3], ade[4])] = "M"+str(id_mention)
            ADE_rel_mention.append(a)
            id_mention+=1
    return ADE_rel_mention, id_mention, ad
	
def extract_modifiers_relation_from_label(m, dict_ade, id_mention, TAC_guess, drug_relation, toks_relation, type_relation, sec_relation, start_relation, len_relation):
    id_relation = 1
    modifiers = []
    relations = []
    m_dict = []
    for ade, drug_r, toks_r, type_r, sec_r, start_r, len_r in zip(TAC_guess.t_ade, drug_relation, toks_relation,type_relation, sec_relation, start_relation, len_relation):
        if len(set(drug_r))==0:
            continue
        if list(set(drug_r))[0]==m:
            arg1 = dict_ade[(ade[0], ade[1], ade[2], ade[3], ade[4])]

            for tok, typ, sec, st, le in zip(toks_r, type_r, sec_r, start_r, len_r): 
                
                modifier_type = typ.split('_')[0]
                relation_type = typ.split('_')[1]
                
                arg2 = "M"+str(id_mention) 
                m_dict.append((str(le), str(st), modifier_type, tok, sec)) 
                id_mention = id_mention + 1
                    
                modifier = (str(le), str(st), modifier_type, tok, sec, arg2)
                
                modifiers.append(modifier)
                relations.append(("RL"+str(id_relation), arg1, arg2, relation_type))

                id_relation = id_relation + 1
    return relations, modifiers