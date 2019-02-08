# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from collections import defaultdict
from pathlib import Path
from typing import Union

from tqdm import tqdm


class Watcher:
    """
    A class that ensures that there are no violations in the structure of mentions.
    """

    def __init__(self):
        self.mentions = 0

    def mentions_closed(self, s: str) -> bool:
        s = s.split('|')
        for x in s:
            if x[0] == '(' and x[-1] != ')':
                self.mentions += 1
            elif x[0] != '(' and x[-1] == ')':
                self.mentions -= 1

        if self.mentions != 0:
            return False
        else:
            return True


def rucoref2conll(path: Union[str, Path], out_path: Union[str, Path]) -> None:
    """
    RuCor corpus files are converted to standard conll format.

    Args:
        path: path to the RuCorp dataset
        out_path: the path to the folder in which the new dataset will be created

    Returns:
        None
    """
    language = 'russian'
    if isinstance(path, str):
        path = Path(path)
    if isinstance(out_path, str):
        out_path = Path(out_path)

    data = {"doc_id": [],
            "part_id": [],
            "word_number": [],
            "word": [],
            "part_of_speech": [],
            "parse_bit": [],
            "lemma": [],
            "sense": [],
            "speaker": [],
            "entiti": [],
            "predict": [],
            "coref": []}

    part_id = '0'
    speaker = 'spk1'
    sense = '-'
    entiti = '-'
    predict = '-'

    tokens_ext = "txt"
    groups_ext = "txt"
    tokens_fname = "Tokens"
    groups_fname = "Groups"

    tokens_path = path.joinpath(".".join([tokens_fname, tokens_ext]))
    groups_path = path.joinpath(".".join([groups_fname, groups_ext]))

    print('[ Convert rucoref corpus into conll format ... ]')
    start = time.time()
    coref_dict = {}
    with groups_path.open("r", encoding='utf8') as groups_file:
        for line in groups_file:
            doc_id, variant, group_id, chain_id, link, shift, lens, content, tk_shifts, attributes, head, hd_shifts = \
                line[:-1].split('\t')

            if doc_id not in coref_dict:
                coref_dict[doc_id] = {'unos': defaultdict(list), 'starts': defaultdict(list), 'ends': defaultdict(list)}

            if len(tk_shifts.split(',')) == 1:
                coref_dict[doc_id]['unos'][shift].append(chain_id)
            else:
                tk = tk_shifts.split(',')
                coref_dict[doc_id]['starts'][tk[0]].append(chain_id)
                coref_dict[doc_id]['ends'][tk[-1]].append(chain_id)

    # Write conll structure
    with tokens_path.open("r", encoding='utf8') as tokens_file:
        k = 0
        doc_name = '0'
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')

            if doc_id == 'doc_id':
                continue

            if doc_id != doc_name:
                doc_name = doc_id
                w = Watcher()
                k = 0

            data['word'].append(token)
            data['doc_id'].append(doc_id)
            data['part_id'].append(part_id)
            data['lemma'].append(lemma)
            data['sense'].append(sense)
            data['speaker'].append(speaker)
            data['entiti'].append(entiti)
            data['predict'].append(predict)
            data['parse_bit'].append('-')

            opens = coref_dict[doc_id]['starts'][shift] if shift in coref_dict[doc_id]['starts'] else []
            ends = coref_dict[doc_id]['ends'][shift] if shift in coref_dict[doc_id]['ends'] else []
            unos = coref_dict[doc_id]['unos'][shift] if shift in coref_dict[doc_id]['unos'] else []
            s = []
            s += ['({})'.format(el) for el in unos]
            s += ['({}'.format(el) for el in opens]
            s += ['{})'.format(el) for el in ends]
            s = '|'.join(s)
            if len(s) == 0:
                s = '-'
                data['coref'].append(s)
            else:
                data['coref'].append(s)

            closed = w.mentions_closed(s)
            if gram == 'SENT' and not closed:
                data['part_of_speech'].append('.')
                data['word_number'].append(k)
                k += 1

            elif gram == 'SENT' and closed:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k = 0
            else:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k += 1

    # Write conll structure in file
    conll = out_path.joinpath(".".join([language, 'v4_conll']))
    with conll.open('w', encoding='utf8') as CoNLL:
        for i in tqdm(range(len(data['doc_id']))):
            if i == 0:
                CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i], data["part_id"][i]))
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                       data["part_id"][i],
                                                                                       data["word_number"][i],
                                                                                       data["word"][i],
                                                                                       data["part_of_speech"][i],
                                                                                       data["parse_bit"][i],
                                                                                       data["lemma"][i],
                                                                                       data["sense"][i],
                                                                                       data["speaker"][i],
                                                                                       data["entiti"][i],
                                                                                       data["predict"][i],
                                                                                       data["coref"][i]))
            elif i == len(data['doc_id']) - 1:
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                       data["part_id"][i],
                                                                                       data["word_number"][i],
                                                                                       data["word"][i],
                                                                                       data["part_of_speech"][i],
                                                                                       data["parse_bit"][i],
                                                                                       data["lemma"][i],
                                                                                       data["sense"][i],
                                                                                       data["speaker"][i],
                                                                                       data["entiti"][i],
                                                                                       data["predict"][i],
                                                                                       data["coref"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
            else:
                if data['doc_id'][i] == data['doc_id'][i + 1]:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                           data["part_id"][i],
                                                                                           data["word_number"][i],
                                                                                           data["word"][i],
                                                                                           data["part_of_speech"][i],
                                                                                           data["parse_bit"][i],
                                                                                           data["lemma"][i],
                                                                                           data["sense"][i],
                                                                                           data["speaker"][i],
                                                                                           data["entiti"][i],
                                                                                           data["predict"][i],
                                                                                           data["coref"][i]))
                    if data["word_number"][i + 1] == 0:
                        CoNLL.write('\n')
                else:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                                                           data["part_id"][i],
                                                                                           data["word_number"][i],
                                                                                           data["word"][i],
                                                                                           data["part_of_speech"][i],
                                                                                           data["parse_bit"][i],
                                                                                           data["lemma"][i],
                                                                                           data["sense"][i],
                                                                                           data["speaker"][i],
                                                                                           data["entiti"][i],
                                                                                           data["predict"][i],
                                                                                           data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i + 1], data["part_id"][i + 1]))

    print('[ End of convertion. Time - {} ]'.format(time.time() - start))


def split_doc(inpath: Union[str, Path], outpath: Union[str, Path]) -> None:
    """
    It splits one large conll file containing the entire RuCorp dataset into many separate documents.

    Args:
        inpath: -
        outpath: -

    Returns:
        None
    """
    language = 'russian'
    if isinstance(inpath, str):
        inpath = Path(inpath)
    if isinstance(outpath, str):
        outpath = Path(outpath)

    # split massive conll file to many little
    print('[ Start of splitting ... ]')
    with inpath.open('r+', encoding='utf8') as f:
        lines = f.readlines()

    set_ends = []
    k = 0

    print('[ Splitting conll document ... ]')
    for i in range(len(lines)):
        if lines[i].startswith('#begin'):
            doc_num = lines[i].split(' ')[2][1:-2]
        elif lines[i] == '#end document\n':
            set_ends.append([k, i, doc_num])
            k = i + 1
    for i in range(len(set_ends)):
        cpath = outpath.joinpath(".".join([str(set_ends[i][2]), language, 'v4_conll']))
        with cpath.open('w', encoding='utf8') as c:
            for j in range(set_ends[i][0], set_ends[i][1] + 1):
                if lines[j] == '#end document\n':
                    c.write(lines[j][:-1])
                else:
                    c.write(lines[j])

    print('[ Splitts {} docs in {} ]'.format(len(set_ends), outpath))
    del lines
    del set_ends
    del k


def get_all_texts_from_tokens_file(tokens_path: Union[str, Path], out_path: Union[str, Path]) -> None:
    """
    Creates file with pure text from RuCorp dataset.

    Args:
        tokens_path: -
        out_path: -

    Returns:
        None

    """
    lengths = {}
    if isinstance(tokens_path, str):
        tokens_path = Path(tokens_path)
    if isinstance(out_path, str):
        out_path = Path(out_path)

    # determine number of texts and their lengths
    with tokens_path.open("r", encoding='utf8') as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                lengths[doc_id] = shift + length
            except ValueError:
                pass

    texts = {doc_id: [' '] * length for (doc_id, length) in lengths.items()}
    # read texts
    with tokens_path.open("r", encoding='utf8') as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                texts[doc_id][shift:shift + length] = token
            except ValueError:
                pass
    for doc_id in texts:
        texts[doc_id] = "".join(texts[doc_id])

    with out_path.open("w", encoding='utf8') as out_file:
        for doc_id in texts:
            out_file.write(texts[doc_id])
            out_file.write("\n")
