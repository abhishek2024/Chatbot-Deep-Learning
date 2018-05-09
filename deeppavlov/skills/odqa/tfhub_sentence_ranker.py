import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.commands.utils import expand_path, get_deeppavlov_root, set_deeppavlov_root
from nltk import sent_tokenize

class TFHUBSentenceRanker:
    def __init__(self):

        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def __call__(self, query_cont, top_k=20):

        '''query_cont is of type List[Tuple[str, List[str]]]'''
        predictions = []
        for el in query_cont:
            query_emb = self.session.run(self.embed([el[0]]))
            cont_embs = self.session.run(self.embed(el[1]))
            scores = (query_emb @ cont_embs.T).squeeze()
            top_ids = np.argsort(scores)[::-1][:top_k]
            predictions.append([el[1][x] for x in top_ids])
        return predictions

def main():

    set_deeppavlov_root({"deeppavlov_root": "/home/leonid/github/DeepPavlov/download"})
    fname = expand_path('Drones.txt')
    with open(fname, 'r') as f:
        data = f.readlines()
    par = list(filter(lambda x: x != '', [el.strip('\n').replace('\ufeff', '') for el in data]))
    text = ' '.join(par)
    sen = sent_tokenize(text)
    pr = TFHUBSentenceRanker()
    predictions = pr([['When is it planned to certify the device?', sen],
                      ['When is it planned to certify the drone?', sen],
                      ['Is sberbank international?', sen]], top_k=5)
    print(predictions)

if __name__ == "__main__":
    main()