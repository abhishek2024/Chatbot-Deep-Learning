## DeepPavlov NER
DeepPavlov NER is an implementation of the deep neural network model for the task of Named Entity Recognition, extended from the model shown in the paper [Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition](https://link.springer.com/chapter/10.1007%2F978-3-319-71746-3_8#enumeration/ "Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition"). In this implementation three sub-networks are employed to fully exploit character-level features, capitalization features as well as word contextual features. DeepPavlov NER was evaluated on Vietnamese, Russian, English and Chinese datasets (VLSP2016, CoNLL2003, NE3, MSRA) and obtained state-of-the-art performances. 

## Training data
To train the neural network, you need to prepare a dataset in the following format:

    EU B-ORG
    rejects O
    the O
    call O
    of O
    Germany B-LOC
    to O
    boycott O
    lamb O
    from O
    Great B-LOC
    Britain I-LOC
    . O
    
    China B-LOC
    says O
    time O
    right O
    for O
    Taiwan B-LOC
    talks O
    . O

    ...

The dataset consists of three text files: train.txt, valid.txt and test.txt (for the training, validation and testing phrases, respectively). Each row has two part: the word and its tag. The sentences are separated by empty lines.


## Parameters Configuration

Here are basic parameters of the model:

- dicts_file: the path to the pickle file containing dictionaries needed for conversation between raw data and indexed data. These dictionaries are:
	- word2id, id2word
	- char2id, id2char
	- tag2id, id2tag
- pretrained_emb: the path to the pre-trained word embedding file
- word_dim: the length of word vectors
- word_hidden_size: the number of units in word LSTM Cell
- char_dim: the length of character vectors
- nb_filters_1: number of output features in the first convolutional layer
- nb_filters_2: number of output features in the second convolutional layer
- cap_dim: the length of capitalization vectors
- cap_hidden_size: the number of units in cap. LSTM Cell
- learning_rate: learning rate to use during training phrase
- drop_out: probability of dropping the hidden state, values in range [0, 1] (0.5 works well in most case)
- nb_epochs: number of epochs in the training phrase
- batch_size: number of sentences in each batch

All these parameters can be easily set in the JSON configuration file:
```json
"dicts_file": "dicts/glove6B100d.iob.l.pkl",
"pretrained_emb": "pretrained_embeddings/glove.6B.100d.txt",
"word_dim": 100,
"word_hidden_size": 300,
"char_dim": 100,
"nb_filters_1": 128,
"nb_filters_2": 150,
"cap_dim": 10,
"cap_hidden_size": 15,
"drop_out": 0.5
```

### Dataset Reader

The dataset reader is a class which reads and parses the data. It returns a dictionary with 
three fields: "train", "test", and "valid". The basic dataset reader is "ner_dataset_reader." 
Bellow are JSON config part for dataset reader:
```json
"dataset_reader": {
    "name": "ner_dataset_reader",
    "data_path": "/home/user/Data/conll2003/"
} 
```

where "name" refers to the basic ner dataset reader class and data_path is the path to the 
folder with three files, namely: "train.txt", "valid.txt", and "test.txt". Each file should
contain data in the format presented in *Training data* section. Each line in the file 
may contain additional information such as POS tags. However, the token must be the first in 
line and NER tag must be the last.

### Dataset

For simple batching and shuffling you can use "basic_dataset". The part of the 
configuration file for the dataset looks like:
 ```json
"dataset": {
    "name": "basic_dataset"
}
```

There is no additional parameters in this part.


## Model Usage

Please see an example of training a NER model and using it for prediction:

```python
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
PIPELINE_CONFIG_PATH = 'configs/ner/ner_conll2003.json'
train_model_from_config(PIPELINE_CONFIG_PATH)
interact_model(PIPELINE_CONFIG_PATH)
```

This example assumes that the working directory is deeppavlov.


## Experiments

DeepPavlov NER was experimented on bellow datasets:
- VLSP2016 [1]: The  Vietnamese  dataset  provided  by  theVietnamese  Language  and  Speech  Processing community [1].
- CoNLL2003 [2]: The  English  dataset  in  the  share task  for  NER  at  Conference  on  Computational  NaturalLanguage Learning, 2003
- Gareev's dataset [3]: The Russian dataset recieved from Ga-reev et al
- MSRA [4]: Due to the difficult of finding an official Chinesedataset  for  the  task  of  NER,  we  decided  to  use  MSRAdataset6.  This  dataset  was  annotated  by  the  NaturalLanguage  Computing  group  within  Microsoft  ResearchAsia.

Performances of the model on these datasets are shown bellow:

| Datasets         	| Precision	| Recall	| F1	|
|-------------------|:---------:|:---------:|:-----:|
| VLSP2016			| 88.61		| 86.54		| 87.57	|
| CoNLL2003			| 90.12		| 91.11		| 90.61	|               |
| Gareev's dataset	| 87.07		| 90.40		| 88.69	|
| MSRA				| 91.99		| 93.92		| 92.95	|

## References
[1] http://vlsp.org.vn

[2] Erik F. Tjong Kim Sang, Fien De Meulder, “Introduction to the conll-2003  shared  task:  Language-independent  named  entity  recognition,”  inProceedings of CoNLL-2003, Edmonton, Canada, 2003, pp. 142–147.

[3] - R.  Gareev,  M.  Tkachenko,  V.  Solovyev,  A.  Simanovsky,  V.  Ivanov,“Introducing baselines for russian named entity recognition,” inCompu-tational Linguistics and Intelligent Text Processing, vol. 7816.   Springer,Berlin, Heidelberg, 2013, pp. 329–342.

[4] - https://www.microsoft.com/en-us/download/details.aspx?id=52531