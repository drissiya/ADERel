from data_utils.tac.tac_corpus import TAC, split_tac_data
from data_utils.n2c2.n2c2_corpus import N2C2, split_n2c2_data
from argparse import ArgumentParser

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from models.loader import Data_Loader
from models.processor import DataProcessor
from models.utils import write_guess_xml_files, write_brat_files

from models.depEmb import DepEmbedding

from models.ajtwgcn.jointModel import AJTWGCNModel
from models.ajtwgcn.loss import JointLossWrapper
from models.ajtwgcn.learner import AJTWGCNLearner
from models.ajtwgcn.inference import AJTWGCNInference
from models.ajtwgcn.extract import tac_extract_guess_relations

from models.twgcn.model import TWGCNModel
from models.twgcn.learner import TWGCNLearner
from models.twgcn.inference import TWGCNInference
from models.twgcn.extract import n2c2_extract_guess_relation

from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict, OrderedDict

from collections import OrderedDict
from pathlib import Path

import os
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import codecs
import random


def get_args():
    parser = ArgumentParser(description="ADERel")

    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--dataset-name', type=str, default='tac',
                        help='tac/n2c2')   
    parser.add_argument('--output_dir', type=str, default='checkpoint')
    
    parser.add_argument('--segment', type=str, default='BIO',
                        help='BIO/BILOU')
    parser.add_argument('--dim-emb', type=int, default=200)
       
    parser.add_argument('--pretrained-models-dir', type=str, default='pretrained_models')


    parser.add_argument('--bert-type', type=str, default='biobert',
                        help='biobert/scibert/bert')
    
    parser.add_argument('--step', type=str, default='train',
                        help='train/test')
    
    parser.add_argument('--vocab-file', type=str, default='vocab.txt')
    parser.add_argument('--config-file', type=str, default='config.json')
    parser.add_argument('--bert-model-file', type=str, default='pytorch_model.bin')

    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
     
    parser.add_argument('--wgcn-hidden-dim', type=int, default=768)
    parser.add_argument('--wgcn-num-layers', type=int, default=1)
    parser.add_argument('--gcn-type', type=str, default='weighted',
                        help='weighted/None')

    parser.add_argument('--attention-type', type=str, default='multi-head',
                        help='multi-head/self/None')
    parser.add_argument('--num-heads', type=int, default=2)    
    
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--weight-decay', type=int, default=0.05)
    
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
       

    args = parser.parse_args()
    return args
	
def main():
    args = get_args()
    args.max_levels = 3

    train_dir_path = os.path.join(args.data_dir, args.dataset_name, "train")
    gold_dir_path = os.path.join(args.data_dir, args.dataset_name, "gold")
    guess_dir_path = os.path.join(args.data_dir, args.dataset_name, "guess")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset_name == "tac":
        train_tac_dir_sentences_path = os.path.join(args.data_dir, args.dataset_name, "TR")
        gold_tac_dir_sentences_path = os.path.join(args.data_dir, args.dataset_name, "TE")

        tac_train = TAC(segment=args.segment,
                        max_level=args.max_levels, 
                        sentence_dir=train_tac_dir_sentences_path, 
                        label_dir=train_dir_path)
        tac_train.load_corpus()

        tac_train, tac_valid = split_tac_data(tac_train)

        tac_test = TAC(segment=args.segment,
                      max_level=args.max_levels, 
                      sentence_dir=gold_tac_dir_sentences_path, 
                      label_dir=gold_dir_path)
        tac_test.load_corpus()

        args.embedding_dep_path =  os.path.join(args.pretrained_models_dir, 'embedding_dep_tac.pkl')
        args.dep2idx_path =  os.path.join(args.pretrained_models_dir, 'dep2idx_tac.pkl')
        dep_feature = DepEmbedding(args, tac_train, tac_test, tac_valid)

        processor = DataProcessor(tac_train, 
                                  tac_test, 
                                  tac_valid)    

    elif args.dataset_name == "n2c2":
        n2c2_train = N2C2(segment = args.segment, label_dir=train_dir_path)
        n2c2_train.load_corpus()

        n2c2_train, n2c2_valid = split_tac_data(n2c2_train)

        n2c2_test = N2C2(segment = args.segment, label_dir=gold_dir_path)
        n2c2_test.load_corpus()


        args.embedding_dep_path =  os.path.join(args.pretrained_models_dir, 'embedding_dep_n2c2.pkl')
        args.dep2idx_path =  os.path.join(args.pretrained_models_dir, 'dep2idx_n2c2.pkl')
        dep_feature = DepEmbedding(args, n2c2_train, n2c2_test, n2c2_valid)

        processor = DataProcessor(n2c2_train, 
                                  n2c2_test, 
                                  n2c2_valid)    

    if not Path(args.embedding_dep_path).is_file():
        dep_feature.train()
    else:
        dep_feature.load()
        
    args.dep_matrix = dep_feature.embedding_dep
    args.dep2idx = dep_feature.dep2idx
    tokenizer = BertTokenizer(vocab_file=os.path.join(args.pretrained_models_dir, args.bert_type, args.vocab_file), 
                                      do_lower_case=False)

    tac_loader = Data_Loader(args, 
                          processor, 
                          tokenizer)
    train_loader, valid_loader, test_loader  = tac_loader.get_all_data_loader(args.dataset_name, print_examples=True)

    args.num_labels = len(processor.get_labels()) + 1
            
    with open(os.path.join(args.pretrained_models_dir, F'label2id_{args.dataset_name}_{args.segment}.pkl'), 'rb') as f: 
        label_map = pkl.load(f)
    label_map = {i: w for w, i in label_map.items()}

    config = BertConfig.from_json_file(os.path.join(args.pretrained_models_dir, args.bert_type, args.config_file))
    tmp_d = torch.load(os.path.join(args.pretrained_models_dir, args.bert_type, args.bert_model_file), map_location=device)
    state_dict = OrderedDict()
    for i in list(tmp_d.keys())[:199]:
        x = i
        if i.find('bert') > -1:
            x = '.'.join(i.split('.')[1:])
        state_dict[x] = tmp_d[i]

    total_steps = len(train_loader) * args.epochs

    if args.dataset_name == "tac":
        model = AJTWGCNModel(args,config,state_dict).to(device)
        loss_fn = JointLossWrapper(2, device)
        print (model)
    elif args.dataset_name == "n2c2":
        model = TWGCNModel(args,config,state_dict).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        print (model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}]

    optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.lr,
                weight_decay=args.weight_decay,
                eps=1e-8
            )
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

    if args.step == "train":
        if args.dataset_name=="tac":
            learner = AJTWGCNLearner(args, 
                                    model, 
                                    optimizer, 
                                    train_loader, 
                                    valid_loader, 
                                    loss_fn, 
                                    device, 
                                    scheduler)

        elif args.dataset_name=="n2c2":
            learner = TWGCNLearner(args, 
                                    model, 
                                    optimizer, 
                                    train_loader, 
                                    valid_loader, 
                                    loss_fn, 
                                    device, 
                                    scheduler)

        os.makedirs(args.output_dir, exist_ok=True)
        learner.train()

    if args.step == "test":
        model_save = F"{args.dataset_name}_model.bin"
        path = F"{args.output_dir}/{model_save}" 
        model.load_state_dict(torch.load(path, map_location=device))

        if args.dataset_name=="tac":
            inference = AJTWGCNInference(args, 
                                        model, 
                                        test_loader, 
                                        label_map, 
                                        device) 
            predicted_labels = inference.predict()

            dict_ade, dict_modifiers, dict_relations = tac_extract_guess_relations(predicted_labels, 
                                                                                  tac_test, 
                                                                                  segment=args.segment,
                                                                                  gold_dir=gold_dir_path)
            os.makedirs(guess_dir_path, exist_ok=True)
            write_guess_xml_files(gold_dir_path, 
                                  guess_dir_path, 
                                  dict_ade,
                                  dict_modifiers,
                                  dict_relations)

        elif args.dataset_name=="n2c2":
            inference = TWGCNInference(args, 
                                        model, 
                                        test_loader, 
                                        label_map, 
                                        device) 
            
            predicted_labels = inference.predict()
            n2c2_test.t_segment_relation = predicted_labels
            dict_ade, dict_modifiers, dict_relations = n2c2_extract_guess_relation(n2c2_test, 
                                                                                  segment=args.segment, 
                                                                                  gold_dir=gold_dir_path)
            write_brat_files(guess_dir_path, 
                            dict_ade, 
                            dict_modifiers, 
                            dict_relations)		
if __name__ == '__main__':
    main()