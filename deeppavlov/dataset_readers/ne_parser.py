import sqlite3
from contextlib import closing
from pathlib import Path
import csv
from typing import List, Tuple, Callable, Iterable, Union, Dict, Optional
from nltk import sent_tokenize, word_tokenize
import pandas as pd

TAG_OUTER = 'O'

NE5_dataset = Path('/data/NER/VectorX/NE5_train')
# NE5_dataset = Path('/data/NER/Academic Datasets/NE5/Collection5')


def get_spans(text: str, tokenizer_or_words: Union[Callable[[str], List[str]], List[str]],
              subs_map: Optional[Dict[str, str]]=None) -> List[Tuple[str, int, int]]:
    if subs_map is None:
        subs_map = {}

    wi = 0
    words = tokenizer_or_words(text) if callable(tokenizer_or_words) else tokenizer_or_words
    word_origs = [subs_map.get(w, w) for w in words]
    words_span = []

    i = 0
    while i < len(text) and wi < len(words):
        if text[i:].startswith(word_origs[wi]):
            words_span.append((words[wi], i, i+len(word_origs[wi])))
            i += len(word_origs[wi])
            wi += 1
            continue
        i += 1

    return words_span


t = 'some \n \ttext\n'
for st, ss, se in get_spans(t, word_tokenize):
    assert st == t[ss: se]

for st, ss, se in get_spans(t, ['some', 'text']):
    assert st == t[ss: se]


def split_sentences_preserve_tags(text: str,
                                  tags_with_spans: Iterable[Tuple[str, int, int, str]],
                                  sentence_tokenizer: Callable[[str], Iterable[str]]):
    tags_with_spans = sorted(tags_with_spans, key=lambda x: x[1])  # excess
    sents = sentence_tokenizer(text)
    #     print(text)
    sents_span = get_spans(text, sentence_tokenizer)
    # sents_span = get_spans(text, sents)

    per_sentence_tags = []
    for orig_s, s_start, s_end in sents_span:
        sent_tags = []
        s = orig_s.strip()
        offset = orig_s.index(s)
        s_start = s_start + offset
        s_end = s_start + len(s)

        for tag, t_start, t_end, value in tags_with_spans:
            if s_start <= t_start <= s_end:
                assert s_start <= t_end <= s_end, 'Entity is in two sentences'
                assert s[t_start - s_start: t_end - s_start] == value, 'Tag value does not equal to sent spans'
                sent_tags.append((tag, t_start - s_start, t_end - s_start, value))
        per_sentence_tags.append((s, sent_tags))
    return per_sentence_tags


def spans_to_bio(word_spans: List[Tuple[str, int, int]], tags: List[Tuple[int, int, str]]) -> List[Tuple[str, str]]:
    """supposed that tags do not overlap

    returns list of pairs of word and its bio-tag"""

    if not tags:
        return [(w[0], TAG_OUTER) for w in word_spans]

    tags = sorted(tags, key=lambda x: x[0])
    for t1, t2 in zip(tags[:-1], tags[1:]):
        assert t1[1] <= t2[0], 'Tags must not overlap'

    bio = []
    for t_start, t_end, tag in tags:
        last_tag = TAG_OUTER
        for w, w_start, w_end in word_spans:
            if t_start <= w_start < t_end or t_start < w_end < t_end:
                prefix = 'I-' if last_tag == tag else 'B-'
                bio.append((w, prefix + tag))
                last_tag = tag
            else:
                bio.append((w, TAG_OUTER))
                last_tag = TAG_OUTER
    return bio


conn = sqlite3.connect(':memory:')

with closing(conn.cursor()) as c:
    c.execute('''CREATE TABLE sentences (article_id INT,
                                         order_in_article INT,
                                         dataset TEXT, 
                                         value TEXT,
                                         annotated INT)''')
    c.execute('''CREATE TABLE tags (sentence_id INT, 
                                    start_index INT, 
                                    end_index INT, 
                                    tag TEXT)''')
    conn.commit()
    print('Created empty database in memory')


def slurp_NE5_annotated_data(folder_path, dataset_name: str = ''):
    NE5_dataset = Path(folder_path)
    all_texts = []
    with closing(conn.cursor()) as c:
        for ann_fn in NE5_dataset.glob('*.ann'):
            #     ann_fn = Path('/data/NER/VectorX/NE5_train/82180290.ann')
            article_id = ann_fn.name[:-len('.ann')]
            text_fn = Path(str(ann_fn)[:-len('.ann')] + '.txt')

            text = text_fn.read_text()
            all_texts.append({'article_id': article_id, 'text': text})

            annotations = []
            with ann_fn.open(encoding='utf8') as f:
                cr = csv.reader(f, delimiter='\t', quotechar='|')
                for row in cr:
                    assert len(row) == 3, row
                    _, tag_info, tag_value = row
                    tag, start_index, end_index = tag_info.split()
                    annotations.append((tag, int(start_index), int(end_index), tag_value))

            try:
                st = split_sentences_preserve_tags(text, annotations, sent_tokenize)

                for order, (s, stags) in enumerate(st):
                    r = c.execute('''INSERT INTO sentences(article_id, order_in_article, dataset, value, annotated) 
                                     VALUES(?,?,?,?,?)''', (article_id, order, dataset_name, s, 1))
                    s_id = r.lastrowid
                    for st, ss, se, sv in stags:
                        c.execute('INSERT INTO tags(sentence_id, start_index, end_index, tag) VALUES (?, ?, ?, ?)',
                                  (s_id, ss, se, st))
                conn.commit()
            except Exception as ex:
                print(f'Exception "{ex}" for file {text_fn}')
                # print(sent_tokenize(text))
                # print('='*80)


def get_unsupervised_data(dataset_name=''):
    with closing(conn.cursor()) as c:
        r = c.execute('''SELECT oid, 
                                article_id, 
                                value
                         FROM sentences
                         WHERE annotated = 0 AND dataset=?''', (dataset_name,))
        ret = []
        for rowid, article_id, text in r.fetchall():
            ret.append({'rowid': rowid, 'article_id': article_id, 'sentence': text})
    return ret


def get_supervised_data(dataset_name=''):
    with closing(conn.cursor()) as c:
        r = c.execute('''SELECT sentences.oid, 
                                sentences.article_id, 
                                sentences.value, 
                                tags.start_index, 
                                tags.end_index, 
                                tags.tag
                         FROM sentences LEFT OUTER JOIN tags ON sentences.oid = tags.sentence_id 
                         WHERE sentences.annotated = 1 and dataset=?''', (dataset_name,))
        d = r.fetchall()

        data = pd.DataFrame(d, columns='sentences.oid sentences.article_id sentences.value tags.start_index tags.end_index tags.tag'.split())

        nltk_subs_map = {"''": '"', '``': '"'}
        sents = []
        for sid, d in data.groupby('sentences.oid'):
            s = d['sentences.value'].iloc[0]
            tags = []
            for _, row in d.iterrows():
                if not pd.isna(row['tags.start_index']):
                    tags.append((int(row['tags.start_index']), int(row['tags.end_index']), row['tags.tag']))

            word_spans = get_spans(s, word_tokenize, nltk_subs_map)
            words, bio_tags = zip(*spans_to_bio(word_spans, tags))
            sents.append((words, bio_tags))

    return sents


def word_indexi_to_spans(text: str, word_spans: List[Tuple[str, int, int]], subs_map=None):
    if subs_map is None:
        subs_map = {}

    wi = 0

    words = [ws[0] for ws in word_spans]
    word_origs = [subs_map.get(w, w) for w in words]
    words_span = []

    i = 0
    while i < len(text) and wi < len(words):
        if text[i:].startswith(word_origs[wi]):
            words_span.append((words[wi], i, i+len(word_origs[wi])))
            i += len(word_origs[wi])
            wi += 1
            continue
        i += 1

    return words_span


def set_annotation(sentence_rowid: int,
                   tokenization: Union[Callable[[str], List[str]], List[str]],
                   selections: List[Tuple[str, int, int]]):
    with closing(conn.cursor()) as c:
        c.execute('''SELECT value, annotated FROM sentences WHERE oid=?''', sentence_rowid)
        orig_sent, is_annotated = c.fetchone()

        assert not is_annotated, f'Error: sentence {sentence_rowid} has already been annotated'

        words_spans = get_spans(orig_sent, tokenization)

        spans_to_insert = []
        for tag, w_start, w_end_incl in selections:
            spans_to_insert.append((sentence_rowid, words_spans[w_start][1], words_spans[w_end_incl][2], tag))

        c.execute('''UPDATE sentences SET annotated=1 WHERE oid=?''', sentence_rowid)
        c.executemany('''INSERT INTO tags (sentence_id, start_index, end_index, tag) VALUES (?, ?, ?, ?)''',
                      spans_to_insert)

        conn.commit()

