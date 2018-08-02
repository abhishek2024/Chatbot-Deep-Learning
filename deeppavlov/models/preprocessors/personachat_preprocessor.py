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


def map_fn(fn, obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [map_fn(fn, o) for o in obj]
    else:
        return fn(obj)


def get_shape(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return (len(obj), *get_shape(obj[0]))
    else:
        return ()


@register('personachat_tokenizer')
class PersonaChatTokenizer(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, **kwargs):
        """ Tokenizes utterances
        Args:
            utterances: list of str

        Returns:
            list of tokens
        """
        return map_fn(word_tokenize, utterances)


@register('personachat_combine_utt_dh')
class PersonaChatCombineUttDH(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, dialog_histories):
        """ Merges utterances and dialog histories

        Args:
            utterances: list of tokens
            dialog_histories: list of tokens

        Returns:
            dialog_history + utterance
        """
        dialog_histories_merged = []
        for dh in dialog_histories:
            dialog_histories_merged.append([])
            for utt in dh:
                dialog_histories_merged[-1].extend(utt)
        # prepend dialog_history to utt
        # TODO: add special token between dialog_history and utt
        full_context = [d + u for u, d in zip(utterances, dialog_histories_merged)]
        return full_context


@register('personachat_merge_personas')
class PersonaChatMergePersonas(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, personas):
        personas_merged = []
        for persona in personas:
            personas_merged.append([])
            for utt in persona:
                personas_merged[-1].extend(utt)
        return personas_merged


@register('personachat_new_vocab')
class PersonachatEmbedder(Estimator):
    # TODO: refactor to merge this code with SQuAD embedder
    def __init__(self, emb_folder, emb_url, save_path, load_path, level='token', *args, **kwargs):
        self.emb_folder = expand_path(emb_folder)
        self.level = level
        self.emb_url = emb_url
        self.emb_file_name = Path(emb_url).name
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)

        self.loaded = False

        self.NULL = "<NULL>"
        self.OOV = "<OOV>"

        self.emb_folder.mkdir(parents=True, exist_ok=True)

        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        if self.load_path.exists():
            self.load()

    def __call__(self, utterances):
        if self.level == 'token':
            result = map_fn(self._get_idx, utterances)
        elif self.level == 'char':
            result = map_fn(self._get_idx, map_fn(list, utterances))
        return result

    def fit(self, x_utterances, y_utterances, personas, y_candidates, *args, **kwargs):
        self.vocab = Counter()
        self.embedding_dict = dict()
        if not self.loaded:
            logger.info('PersonachatEmbedder: fitting with {}s'.format(self.level))
            data = x_utterances + y_utterances + personas
            candidates_merged = []
            for candidates in y_candidates:
                candidates_merged.append([])
                for utt in candidates:
                    candidates_merged[-1].extend(utt)
            data = list(data) + candidates_merged

            if self.level == 'token':
                for line in tqdm(data):
                    for token in line:
                        self.vocab[token] += 1
            elif self.level == 'char':
                for line in tqdm(data):
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


@register('personachat_cut_len')
class PersonaChatCutLen(Component):
    def __init__(self, token_limit, char_limit=None, level='token', *args, **kwargs):
        self.token_limit = token_limit
        self.char_limit = char_limit
        self.level = level
        if level == 'char' and self.char_limit is None:
            raise RuntimeError("char lvl is set but char_limit is not set")

    def __call__(self, batch):
        if self.level == 'token':
            if len(get_shape(batch)) == 3:
                utt_idxs = np.zeros([len(batch), len(batch[0]), self.token_limit], dtype=np.int32)
                for b, utterances in enumerate(batch):
                    for i, utt in enumerate(utterances):
                        for j, token in enumerate(utt[-self.token_limit:]):
                            utt_idxs[b, i, j] = token
            elif len(get_shape(batch)) == 2:
                utt_idxs = np.zeros([len(batch), self.token_limit], dtype=np.int32)
                for i, utt in enumerate(batch):
                    for j, token in enumerate(utt[-self.token_limit:]):
                        utt_idxs[i, j] = token
            else:
                raise RuntimeError("Unsupported batch shape")

        elif self.level == 'char':
            if len(get_shape(batch)) == 4:
                utt_idxs = np.zeros([len(batch), len(batch[0]), self.token_limit, self.char_limit], dtype=np.int32)
                for b, utterances in enumerate(batch):
                    for i, utt in enumerate(utterances):
                        for j, token in enumerate(utt[-self.token_limit:]):
                            for k, char in enumerate(token[:self.char_limit]):
                                utt_idxs[b, i, j, k] = char
            elif len(get_shape(batch)) == 3:
                utt_idxs = np.zeros([len(batch), self.token_limit, self.char_limit], dtype=np.int32)
                for i, utt in enumerate(batch):
                    for j, token in enumerate(utt[-self.token_limit:]):
                        for k, char in enumerate(token[:self.char_limit]):
                            utt_idxs[i, j, k] = char
            else:
                raise RuntimeError("Unsupported batch shape")

        return utt_idxs

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


@register_metric('personachat_loss_tf')
def personachat_loss_tf(y_true, metrics):
    loss = list(map(lambda x: x['loss'], metrics))
    return float(-np.mean(loss))


@register_metric('personachat_perplexity_tf')
def personachat_perplexity_tf(y_true, metrics):
    return np.exp(-personachat_loss_tf(y_true, metrics))


@register_metric('personachat_hits@1_tf')
def personachat_hits1_tf(y_true, metrics):
    score = list(map(lambda x: x['hits@1'], metrics))
    return float(np.mean(score))
