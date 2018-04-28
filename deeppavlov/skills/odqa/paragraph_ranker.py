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

    def __call__(self, query_cont, top_k=1):
        num_cont = [len(el[1]) for el in query_cont]
        num_cont_cum =  np.cumsum([0] + num_cont)
        bs = sum(num_cont)
        x1 = 99999 * np.ones((bs, 336), dtype=int)
        x1_len = np.zeros(bs)
        x2 = 99999 * np.ones((bs, 3566), dtype=int)
        x2_len = np.zeros(bs)
        for i in range(len(query_cont)):
            query = word_tokenize(query_cont[i][0].lower())
            query = [self.word_dict.get(el) for el in query if self.word_dict.get(el) is not None]
            len_q = min(336, len(query))
            for j in range(num_cont[i]):
                x1_len[num_cont_cum[i]+j] = len_q
                x1[num_cont_cum[i]+j, :len_q] = query[:len_q]

            for j in range(num_cont[i]):
                cont = word_tokenize(query_cont[i][1][j].lower())
                cont = [self.word_dict.get(el) for el in cont if self.word_dict.get(el) is not None]
                len_c = min(3566, len(cont))
                x2_len[num_cont_cum[i]+j] = len_c
                x2[num_cont_cum[i]+j, :len_c] = cont[:len_c]
        batch = {'query': x1, 'query_len': x1_len, 'doc': x2, 'doc_len': x2_len}
        predictions = self.model.predict(batch)
        pred_g = []
        for i in range(len(num_cont_cum)-1):
                pred_g.append(predictions[num_cont_cum[i]: num_cont_cum[i+1]].squeeze().argsort()[::-1])
        ans = []
        for i in range(len(query_cont)):
            num_top = min(top_k, num_cont[i])
            ans.append([query_cont[i][1][el] for el in pred_g[i][:num_top]])
        return ans

    def _build_dict(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
        data = [el.strip('\n').split(' ') for el in data]
        word_dict = {el[0]: el[1] for el in data}
        return word_dict

    def _margin_loss(self, y_true, y_pred):
        y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
        y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
        loss = K.maximum(0., 0.1 + y_neg - y_pos)
        return K.mean(loss)


def main():

    set_deeppavlov_root({"deeppavlov_root": "/home/leonid/github/DeepPavlov/download"})
    # download_decompress("http://lnsigo.mipt.ru/export/deeppavlov_data/sber_squad_ranking_arci_40.tar.gz",
    #                     get_deeppavlov_root())
    fname = expand_path('test_data1.txt')
    with open(fname, 'r') as f:
        test_data = f.readlines()
    test_data = [el.strip('\n').split('\t')[1:3] for el in test_data]
    queries = [el[0] for el in test_data]
    conts = [el[1] for el in test_data]
    q_c_dict = {el: [] for el in set(queries)}
    for el in zip(queries, conts):
        q_c_dict[el[0]].append(el[1])
    pr = ParagraphRanker("sber_squad_ranking_arci_40")
    predictions = pr(list(q_c_dict.items()), top_k=2)
    # print(np.argmax(np.reshape(predictions, (13, 10)), axis=1))
    print(predictions)

if __name__ == "__main__":
    main()