import numpy as np

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.skills.odqa.basic_neural_context_encoder import BasicNeuralContextEncoder

iterator = SQLiteDataIterator(load_path='/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki.db',
                              shuffle=False)
encoder = BasicNeuralContextEncoder(load_path='/media/olga/Data/projects/DeepPavlov/download/bnr/model')
SAVE_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/chunk_vectors_{}'

all_vectors = []

i = 0
j = 0
for docs, _ in iterator.gen_batches(batch_size=1000):
    if i == 1000:
        stacked = np.concatenate([all_vectors], axis=1)
        np.save(SAVE_PATH.format(j), stacked)
        all_vectors.clear()
        j += 1
        i = 0
    batch_vectors = encoder(docs)
    all_vectors.append(batch_vectors)
    i += 1

np.save(SAVE_PATH, all_vectors)

