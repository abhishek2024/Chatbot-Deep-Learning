import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from nltk import sent_tokenize

from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.commands.utils import expand_path, get_deeppavlov_root, set_deeppavlov_root
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('sentence_ranker')
class TFHUBSentenceRanker(Component):
    def __init__(self, top_k=20, **kwargs):
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.top_k = top_k

    def __call__(self, query_cont):
        '''query_cont is of type List[Tuple[str, List[str]]]'''
        predictions = []
        for el in query_cont:
            query_emb = self.session.run(self.embed([el[0]]))
            cont_embs = self.session.run(self.embed(el[1]))
            scores = (query_emb @ cont_embs.T).squeeze()
            top_ids = np.argsort(scores)[::-1][:self.top_k]
            predictions.append([el[1][x] for x in top_ids])
        res = [' '.join(sentences) for sentences in predictions]
        return res


def main():
    set_deeppavlov_root({"deeppavlov_root": "/home/leonid/github/DeepPavlov/download"})
    fname = expand_path('Drones.txt')
    with open(fname, 'r') as f:
        data = f.readlines()
    par = list(filter(lambda x: x != '', [el.strip('\n').replace('\ufeff', '') for el in data]))
    text = ' '.join(par)
    sen = sent_tokenize(text)
    pr = TFHUBSentenceRanker(top_k=5)
    predictions = pr([['When is it planned to certify the device?', sen],
                      ['When is it planned to certify the drone?', sen],
                      ['Is sberbank international?', sen]])
    print(predictions)


if __name__ == "__main__":
    main()
