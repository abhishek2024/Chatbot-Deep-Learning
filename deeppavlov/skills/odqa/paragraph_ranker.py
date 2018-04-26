import numpy as np
from keras.layers import Lambda
import keras.backend as K
from keras.models import load_model
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.commands.utils import expand_path, get_deeppavlov_root, set_deeppavlov_root
from nltk import word_tokenize

class ParagraphRanker:
    def __init__(self, load_path):

        self.model = load_model(str(expand_path(load_path) / "model.hdf5"),
                                custom_objects={"_margin_loss": self._margin_loss})
        self.word_dict = self._build_dict(str(expand_path(load_path) / "word_dict.txt"))

    def __call__(self, query_cont):
        bs = len(query_cont)
        x1 = 99999 * np.ones((bs, 336), dtype=int)
        x1_len = np.zeros(bs)
        x2 = 99999 * np.ones((bs, 3566), dtype=int)
        x2_len = np.zeros(bs)
        for i in range(bs):
            query = word_tokenize(query_cont[i][0].lower())
            query = [self.word_dict.get(el) for el in query if self.word_dict.get(el) is not None]
            len_q = min(336, len(query))
            x1_len[i] = len_q
            x1[i, :len_q] = query[:len_q]

            cont = word_tokenize(query_cont[i][1].lower())
            cont = [self.word_dict.get(el) for el in cont if self.word_dict.get(el) is not None]
            len_c = min(3566, len(cont))
            x2_len[i] = len_c
            x2[i, :len_c] = cont[:len_c]
        batch = {'query': x1, 'query_len': x1_len, 'doc': x2, 'doc_len': x2_len}
        return self.model.predict(batch)

    def _build_dict(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
        data = [el.strip('\n').split(' ') for el in data]
        word_dict = {el[0]: el[1] for el in data}
        return word_dict

    def _margin_loss(self, y_true, y_pred):
        y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
        y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
        loss = K.maximum(0., 0.1 + y_neg - y_pos)
        return K.mean(loss)


def main():

    set_deeppavlov_root({"deeppavlov_root": "/home/leonid/github/DeepPavlov/download"})
    download_decompress("http://lnsigo.mipt.ru/export/deeppavlov_data/sber_squad_ranking_arci_40.tar.gz",
                        get_deeppavlov_root())
    fname = expand_path('test_data.txt')
    with open(fname, 'r') as f:
        test_data = f.readlines()
    test_data = [el.strip('\n').split('\t')[1:3] for el in test_data]
    pr = ParagraphRanker("sber_squad_ranking_arci_40")
    predictions = pr(test_data)
    print(np.argmax(np.reshape(predictions, (13, 10)), axis=1))

if __name__ == "__main__":
    main()