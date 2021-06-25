from data_utils.tac.drug_label import xml_files
from xml.etree import ElementTree
import torch.optim as optim
import torch.nn as nn
import os

def trim_sequence(prediction, true_set, label_mappers):
    predict_lines = []
    for pred, true in zip(prediction, true_set):
        p_label = []
        for p, t in zip(pred, true):
            if p == 0: 
                l = 'O'
                p_label.append(l)
                continue
            l= label_mappers[p]
            tr= label_mappers[t]
            if tr == 'X': continue
            if l == '[CLS]': continue
            if l == '[SEP]': continue
            if l == 'X': l = 'O'
            p_label.append(l)
        predict_lines.append(p_label)
    return predict_lines
	
def trim(level, preds, valied_lenght):
    final_predict = []
    target = []
    l1 = level.tolist()
    p1 = preds.tolist()
    for idx, (p, t) in enumerate(zip(p1, l1)):
        final_predict.append(p[: valied_lenght[idx]])
        target.append(t[: valied_lenght[idx]])
    return final_predict, target
	
def write_guess_xml_files(gold_xml_dir, 
                          guess_xml_dir, 
                          dict_ade, 
                          dict_modifiers, 
                          dict_relations):
    guess_files = xml_files(gold_xml_dir)
    for key, value in zip(guess_files.keys(), guess_files.values()):
        root = ElementTree.parse(value).getroot()
        root.remove(root[1])
        root.remove(root[2])
        root.remove(root[-1])
        Mentions = ElementTree.SubElement(root, "Mentions")
        for m in dict_ade[key]:
            ElementTree.SubElement(Mentions, "Mention", id=m[5], section=m[4], type=m[2], start=m[1], len=m[0], str=m[3])
        for m in dict_modifiers[key]:
            ElementTree.SubElement(Mentions, "Mention", id=m[5], section=m[4], type=m[2], start=m[1], len=m[0], str=m[3])

        Relations = ElementTree.SubElement(root, "Relations")
        for m in dict_relations[key]:
            ElementTree.SubElement(Relations, "Relation", id=m[0], type=m[3], arg1=m[1], arg2=m[2])

        Reactions = ElementTree.SubElement(root, "Reactions")
        tree = ElementTree.ElementTree(root)
        tree.write(os.path.join(guess_xml_dir, key + '.xml'), encoding="utf-8")


def write_brat_files(guess_dir, 
                     dict_ade, 
                     dict_modifiers, 
                     dict_relations):
    for key in os.listdir(guess_dir):
        if key.endswith('.txt'):
            key = key.replace('.txt', '')
            file = open(os.path.join(guess_dir, key + '.ann'),"w") 

            for m in dict_ade[key]:
                file.write(m[4] + '\t' + m[2] + ' ' + m[1] + ' ' + m[0] + '\t' + m[3] + '\n')

            for m in dict_modifiers[key]:
                file.write(m[4] + '\t' + m[2] + ' ' + m[1] + ' ' + m[0] + '\t' + m[3] + '\n')

            for m in dict_relations[key]:

                file.write(m[0] + '\t' + m[3] + ' Arg1:' + m[2] + ' Arg2:' + m[1] + '\n')
            file.close()