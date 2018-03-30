from collections import Counter
from pathlib import Path
import pickle

from nltk import word_tokenize
import numpy as np
from tqdm import tqdm

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

from deeppavlov.core.common.metrics_registry import register_metric

logger = get_logger(__name__)

@register('personachat_tokenizer')
class PersonaChatTokenizer(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, dialog_histories=None, personas=None, **kwargs):
        """ Tokenizes utterances and agent personality sentences.
            Merges personality sentences into one text.

        Args:
            utterances:
            personas: list of sentences, which characterize personality
            **kwargs:

        Returns:
            tokenized utterances and
        """
        utt_tokenized = [word_tokenize(u) for u in utterances]
        if dialog_histories is not None:
            dialog_histories_tok = list(map(lambda x: word_tokenize(' '.join(x) if isinstance(x, list) else x), dialog_histories))
            # prepend dialog_history to utt
            # TODO: add special token between them
            utt_tokenized = [d + u for u, d in zip(utt_tokenized, dialog_histories_tok)]
        if personas is not None:
            personas_tokenized = list(map(lambda x: word_tokenize(' '.join(x) if isinstance(x, list) else x), personas))
            return utt_tokenized, personas_tokenized
        return utt_tokenized


@register('personachat_vocab')
class PersonachatEmbedder(Estimator):
    # TODO: refactor to merge this code with SQuAD embedder
    def __init__(self, emb_folder, emb_url, save_path, load_path,
                 x_len_limit, persona_len_limit, y_len_limit, char_limit, level='token', *args, **kwargs):
        self.emb_folder = expand_path(emb_folder)
        self.level = level
        self.emb_url = emb_url
        self.emb_file_name = Path(emb_url).name
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.x_len_limit = x_len_limit
        self.persona_len_limit = persona_len_limit
        self.y_len_limit = y_len_limit
        self.char_limit = char_limit
        self.loaded = False

        self.NULL = "<NULL>"
        self.OOV = "<OOV>"

        self.emb_folder.mkdir(parents=True, exist_ok=True)

        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        if self.load_path.exists():
            self.load()

    def __call__(self, utterances, personas=None):
        seq_len_limit = self.x_len_limit if personas is not None else self.y_len_limit
        if self.level == 'token':
            utt_idxs = np.zeros([len(utterances), seq_len_limit], dtype=np.int32)
            for i, utt in enumerate(utterances):
                for j, token in enumerate(utt[-seq_len_limit:]):
                    utt_idxs[i, j] = self._get_idx(token)

            if personas is None:
                return utt_idxs

            per_idxs = np.zeros([len(personas), self.persona_len_limit], dtype=np.int32)
            for i, persona in enumerate(personas):
                for j, token in enumerate(persona[:self.persona_len_limit]):
                    per_idxs[i, j] = self._get_idx(token)

        elif self.level == 'char':
            utt_idxs = np.zeros([len(utterances), seq_len_limit, self.char_limit], dtype=np.int32)
            for i, utt in enumerate(utterances):
                for j, token in enumerate(utt[-seq_len_limit:]):
                    for k, char in enumerate(token[:self.char_limit]):
                        utt_idxs[i, j, k] = self._get_idx(char)

            if personas is None:
                return utt_idxs

            per_idxs = np.zeros([len(personas), self.persona_len_limit, self.char_limit], dtype=np.int32)
            for i, persona in enumerate(personas):
                for j, token in enumerate(persona[:self.persona_len_limit]):
                    for k, char in enumerate(token[:self.char_limit]):
                        per_idxs[i, j, k] = self._get_idx(char)

        return utt_idxs, per_idxs

    def fit(self, x_utterances, y_utterances, personas, *args, **kwargs):
        self.vocab = Counter()
        self.embedding_dict = dict()
        if not self.loaded:
            logger.info('PersonachatEmbedder: fitting with {}s'.format(self.level))
            if self.level == 'token':
                for line in tqdm(x_utterances + y_utterances + personas):
                    for token in line:
                        self.vocab[token] += 1
            elif self.level == 'char':
                for line in tqdm(x_utterances + y_utterances + personas):
                    for token in line:
                        for c in token:
                            self.vocab[c] += 1
            else:
                raise RuntimeError("PersonachatEmbedder::fit: Unknown level: {}".format(self.level))

            with (self.emb_folder / self.emb_file_name).open('r') as femb:
                emb_voc_size, self.emb_dim = map(int, femb.readline().split())
                for line in tqdm(femb, total=emb_voc_size):
                    line_split = line.strip().split(' ')
                    word = line_split[0]
                    vec = np.array(line_split[1:], dtype=float)
                    if len(vec) != self.emb_dim:
                        continue
                    if word in self.vocab:
                        self.embedding_dict[word] = vec

            self.token2idx_dict = {token: idx for idx,
                                             token in enumerate(self.embedding_dict.keys(), 2)}
            self.idx2token = [self.NULL, self.OOV] + list(self.embedding_dict.keys())
            self.token2idx_dict[self.NULL] = 0
            self.token2idx_dict[self.OOV] = 1
            self.embedding_dict[self.NULL] = [0. for _ in range(self.emb_dim)]
            self.embedding_dict[self.OOV] = [0. for _ in range(self.emb_dim)]
            idx2emb_dict = {idx: self.embedding_dict[token]
                            for token, idx in self.token2idx_dict.items()}
            self.emb_mat = np.array([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

    def load(self, *args, **kwargs):
        logger.info('PersonachatEmbedder: loading saved {}s vocab from {}'.format(self.level, self.load_path))
        self.emb_dim, self.emb_mat, self.token2idx_dict, self.idx2token = pickle.load(self.load_path.open('rb'))
        self.loaded = True

    def save(self, *args, **kwargs):
        logger.info('PersonachatEmbedder: saving {}s vocab to {}'.format(self.level, self.save_path))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((self.emb_dim, self.emb_mat, self.token2idx_dict, self.idx2token), self.save_path.open('wb'))

    def _get_idx(self, el):
        """ Returns idx for el (token or char).

        Args:
            el: token or character

        Returns:
            idx in vocabulary
        """
        for e in (el, el.lower(), el.capitalize(), el.upper()):
            if e in self.token2idx_dict:
                return self.token2idx_dict[e]
        return 1


@register('personachat_postprocessor')
class PersonaChatPostprocessor(Component):
    def __init__(self, idx2token, NULL, *args, **kwargs):
        self.idx2token = idx2token
        self.NULL = NULL

    def __call__(self, idxs, **kwargs):
        """ Converts predicted tokens ids to tokens.
        """
        tokens = []
        for utt in idxs:
            tokens.append([])
            for idx in utt:
                token = self.idx2token[idx]
                if token == self.NULL:
                    break
                tokens[-1].append(token)
        tokens = [' '.join(utt) for utt in tokens]
        
        return tokens


@register_metric('personachat_loss')
def personachat_loss(y_true, y_predicted):
    voc_size = y_predicted[0].shape[-1]
    y_predicted = np.array(y_predicted).reshape(-1, voc_size)
    y_true = np.array(y_true).reshape((-1,))
    #loss = np.sum(np.log(y_predicted[:, np.arange(y_true.shape[1]), y_true][0,:,:]) * (y_true > 0)) / np.sum(y_true>0)
    from sklearn.metrics import log_loss
    loss = -log_loss(y_true=y_true,
                     y_pred=y_predicted,
                     sample_weight=(y_true>0),
                     labels=np.arange(voc_size),
                     eps=1e-06)
    loss = float(loss)
    return loss if len(y_true) > 0 else 0.0


@register_metric('personachat_perplexity')
def personachat_perplexity(y_true, y_predicted):
    return np.exp(-personachat_loss(y_true, y_predicted))