# An attentive joint model with transformer-based weighted graph convolutional network for extracting adverse drug event relation 
This repository contains the source code of our proposed [ADERel system](https://www.sciencedirect.com/science/article/abs/pii/S1532046421002975). Firstly, it formulates the ADE relation extraction task as an N-level sequence labelling so as to model the complex relations in different levels and capture greater interaction between relations. Then, it exploits our neural joint model to process the N-level sequences jointly. The joint model leverages the contextual and structural information by adopting a shared representation that combines a bidirectional encoder representation from transformers (BERT) and our proposed weighted GCN (WGCN). The latter assigns a score to each dependency edge within a sentence so as to capture rich syntactic features and determine the most influential edges for extracting ADE relations. Finally, the system employs a multi-head attention to exchange boundary knowledge across levels. We evaluate ADERel on two benchmark datasets from TAC 2017 and n2c2 2018 shared tasks. More details are provided [here](https://github.com/drissiya/ADERel/tree/main/data).


## Requirements

1. Install all dependencies needed to run this repository:

```
$ pip install -r requirements.txt
```

2. Download the pre-trained model files used in our experiments. They include BERT, BioBERT and SciBERT:

```
$ bash pretrained_models/download.sh
```


## Quick start


1. Train the model:

```
$ python main.py \
  --dataset-name "tac" \
  --step "train" \
  --segment "BIO" \
  --bert-type "biobert" \
  --attention-type "multi-head" \
  --gcn-type "weighted" \
  --epochs 10 \
```

2. Make predictions:

```
$ python main.py \
  --dataset-name "tac" \
  --step "test" \
  --segment "BIO" \
  --bert-type "biobert" \
  --attention-type "multi-head" \
  --gcn-type "weighted" \
  --epochs 10 \
```

After the program is finished, the guess xml files will be generated in the data/ADR/TAC/guess_xml folder for TAC 2017 dataset. 

3. To evaluate ADERel on TAC 2017 dataset, run evaluate.py file as follows:

```
$ python evaluate.py "data/ADR/TAC/gold" "data/ADR/TAC/guess"
```

The same procedure can be used for n2c2 2018 dataset.

## Citation 

```
@article{Elallaly2021AnAJ,
	doi = {10.1016/j.jbi.2021.103968},
	year = 2022,
	volume = {125},
	pages = {103968},
  title={An attentive joint model with transformer-based weighted graph convolutional network for extracting adverse drug event relation.},
  author={Ed-drissiya El-allaly and Mourad Sarrouti and Noureddine En-nahnahi and Said Ouatik El Alaoui},
  journal={Journal of biomedical informatics},
}
```

## Acknowledgements

We are also grateful to the TAC 2017 ADR challenge organizers and the n2c2 challenge organizers who provided us the datasets used to evaluate this work.

