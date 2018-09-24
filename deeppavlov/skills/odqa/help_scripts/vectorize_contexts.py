import numpy as np

from deeppavlov.dataset_iterators.sqlite_iterator import SQLiteDataIterator
from deeppavlov.skills.odqa.basic_neural_context_encoder import BasicNeuralContextEncoder

iterator = SQLiteDataIterator(load_path='/media/olga/Data/projects/DeepPavlov/download/odqa/enwiki_full_chunk.db')
encoder = BasicNeuralContextEncoder(load_path='/media/olga/Data/projects/DeepPavlov/download/bnr/model')
SAVE_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/chunk_vectors'

all_vectors = np.empty(shape=(1, 512))

i = 0
j = 0
for docs, _ in iterator.gen_batches(batch_size=100):
    if i == 10000:
        np.save(SAVE_PATH.format(j), all_vectors)
        all_vectors = np.empty(shape=(1, 512))
        j += 1
        i = 0
    batch_vectors = encoder(docs)
    all_vectors = np.append(all_vectors, batch_vectors, axis=0)
    print(all_vectors.shape)
    i += 1

# print(type(all_vectors))
# print(all_vectors.shape)
np.save(SAVE_PATH, all_vectors)
# a = np.load('/media/olga/Data/projects/DeepPavlov/download/odqa/chunk_vectors.npy')
# print(a)

