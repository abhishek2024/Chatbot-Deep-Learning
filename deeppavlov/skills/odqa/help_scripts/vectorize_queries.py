import numpy as np

from deeppavlov.skills.odqa.basic_neural_query_encoder import BasicNeuralQueryEncoder

encoder = BasicNeuralQueryEncoder(load_path='/media/olga/Data/projects/DeepPavlov/download/bnr/model')
SAVE_PATH = '/media/olga/Data/projects/DeepPavlov/download/odqa/query_vectors'


queries = ['Hello world 1', 'Hello world 2']
all_vectors = np.concatenate([encoder(query) for query in queries])


# print(type(all_vectors))
# print(all_vectors.shape)
np.save(SAVE_PATH, all_vectors)
# a = np.load('/media/olga/Data/projects/DeepPavlov/download/odqa/chunk_vectors.npy')
# print(a)

