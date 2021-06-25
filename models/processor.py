import sys
import numpy as np
from pathlib import Path
import pickle as pkl
import codecs


	
class InputExample(object):
    def __init__(self, guid, text_a, head, label, dep):
        self.guid = guid
        self.text_a = text_a
        self.head = head
        self.label = label
        self.dep = dep


class DataProcessor(object):
    def __init__(self, train_data, test_data, dev_data=None):
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

    def get_train_examples(self):
        return self._create_example(self.read_data(self.train_data), "train")

    def get_test_examples(self):
        return self._create_example(self.read_data(self.test_data), "test")
    
    def get_valid_examples(self):
        return self._create_example(self.read_data(self.dev_data), "valid")

    def get_labels(self):
        l1 = set([a for l in self.train_data.t_segment_relation for i in l for a in i])
        l2 = []
        if self.dev_data is not None:
            l2 = set([a for l in self.dev_data.t_segment_relation for i in l for a in i])
        l3 = set([a for l in self.test_data.t_segment_relation for i in l for a in i])
        return set(list(l1)+list(l2)+list(l3) + ['X', '[CLS]', '[SEP]'])

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines): 
            guid  = "%s-%s" % (set_type, i)
            instance = line[0]
            label = line[1] 
            head = line[2] 
            dep = line[3]  
            examples.append(InputExample(guid=guid, text_a=instance, head=head, label=label, dep=dep))
        return examples

    def read_data(self, tac_set):
        lines  = []
        for i in range(len(tac_set.t_sentence_relation)):
            instance = tac_set.t_sentence_relation[i]
            label = tac_set.t_segment_relation[i]
            head = tac_set.heads[i]
            dep = tac_set.deps[i]
            lines.append([instance, label, head, dep])
        return lines

