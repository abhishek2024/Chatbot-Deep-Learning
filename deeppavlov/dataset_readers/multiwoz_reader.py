# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Union, Tuple

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('multiwoz_reader')
class MultiWOZDatasetReader(DatasetReader):
    """
    # TODO: add docs
    """

    url = 'http://files.deeppavlov.ai/datasets/kvret_public.tar.gz'
    url = None

    DATA_FILES = ['data.json', 'dialogue_acts.json', 'valListFile.json',
                  'testListFile.json',
                  'attraction_db.json', 'bus_db.json',
                  'hospital_db.json', 'hotel_db.json', 'police_db.json',
                  'restaurant_db.json', 'taxi_db.json', 'train_db.json']
    PREPROCESSED = ['data_prep.json']

    @classmethod
    @overrides
    def read(self, data_path: str, domains: List[str] = None,
             dialogs: bool = False) -> Dict[str, List]:
        """
        Parameters:
            data_path: path to save data
            dialogs: flag indices whether to output list of turns or list of dialogs

        Returns:
            dictionary with ``'train'``, ``'valid'`` and ``'test'``.
            Each fields is a list of tuples ``(x_i, y_i)``.
        """
        if dialogs:
            raise NotImplementedError("dialogs=True is not implemented")

        if not all(Path(data_path, f).exists() for f in self.DATA_FILES):
            log.info('[downloading multiwoz from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data_file = Path(data_path, 'data.json')
        act_data_file = Path(data_path, 'dialogue_acts.json')
        databases = {fname[:-8]: Path(data_path, fname)
                     for fname in self.DATA_FILES if fname.endswith('_db.json')}
        print(databases)

        if not all(Path(data_path, f).exists() for f in self.PREPROCESSED):
            log.info('[preprocessing multiwoz]')
            prep_data = self.preprocess_data(data_file, act_data_file, databases)
            with open(Path(data_path, 'data_prep.json'), 'wt') as fout:
                json.dump(prep_data, fout)

        data_file = Path(data_path, 'data_prep.json')

        valid_ids = self._read_indices(Path(data_path, 'valListFile.json'))
        test_ids = self._read_indices(Path(data_path, 'testListFile.json'))
        data = {
            'train': self._read_dialogues(data_file, dialogs=dialogs,
                                          exclude_ids=valid_ids + test_ids,
                                          domains=domains),
            'valid': self._read_dialogues(data_file, dialogs=dialogs, ids=valid_ids,
                                          domains=domains),
            'test':  self._read_dialogues(data_file, dialogs=dialogs, ids=test_ids,
                                          domains=domains)
        }
        return data

    @staticmethod
    def _read_indices(file_path):
        return [ln.strip() for ln in open(file_path, 'rt')]

    @classmethod
    def _read_dialogues(cls, file_path, dialogs=False, ids=None, exclude_ids=[],
                        domains=None):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(file_path))

        dialog_gen = cls._iter_file(file_path, ids=ids, exclude_ids=exclude_ids,
                                    domains=domains)
        utterances, responses, dialog_indices = cls._get_turns(dialog_gen,
                                                               with_indices=True)

        data = list(map(cls._format_turn, zip(utterances, responses)))

        if dialogs:
            return [data[idx['start']:idx['end']] for idx in dialog_indices]
        return data

    @staticmethod
    def _format_turn(turn):
        x = {'text': turn[0]['text'],
             'dialog_id': turn[0]['dialog_id']}
        if turn[0].get('episode_done') is not None:
            x['episode_done'] = turn[0]['episode_done']
        y = {'text': turn[1]['text'],
             'domains': turn[1]['domains'],
             'x_tags': turn[0]['tags']}
        return (x, y)

    @staticmethod
    def _check_dialog(dialog):
        expecting_user = True
        for turn in dialog:
            is_user = bool(turn['metadata'] == {})
            if expecting_user and not is_user:
                log.debug(f"Turn is expected to be by user, but"
                          f" it contains metadata = {turn['metadata']}")
                return False
            elif not expecting_user and is_user:
                log.debug(f"Turn is expected to be by system, but"
                          f" it contains empty metadata")
                return False
            expecting_user = not expecting_user
        if not expecting_user:
            log.debug("Last turn is expected to be by system")
            return False
        return True

    @classmethod
    def _iter_file(cls, file_path, ids=None, exclude_ids=[], domains=None):
        with open(file_path, 'rt', encoding='utf8') as f:
            data = json.load(f)
        for dialog_id, sample in data.items():
            if (dialog_id in exclude_ids) or (ids is not None and dialog_id not in ids):
                continue
            dialog_domains = [d for d, val in sample['goal'].items()
                              if (val != {}) and (val not in ['topic', 'message'])]
            if domains is not None and dialog_domains != domains:
                break
            dialog = sample['log']
            if cls._check_dialog(dialog):
                yield dialog_id, dialog, dialog_domains
            else:
                log.warn(f"Skipping dialogue with uuid={dialog_id}: wrong format.")

    @staticmethod
    def _get_turns(data, with_indices=False):
        utterances, responses, dialog_indices = [], [], []
        for dialog_id, dialog, domains in data:
            for i, turn in enumerate(dialog):
                replica = {'text': turn['tokens']}
                if i == 0:
                    replica['episode_done'] = True
                is_user = bool(turn['metadata'] == {})
                if is_user:
                    replica['dialog_id'] = dialog_id
                    replica['tags'] = turn['tags']
                    utterances.append(replica)
                else:
                    replica['domains'] = domains
                    responses.append(replica)

            # if last replica was by driver
            if len(responses) != len(utterances):
                utterances[-1]['end_dialogue'] = False
                responses.append({'text': '', 'end_dialogue': True})

            responses[-1]['text'] += ['END_OF_DIALOGUE']

            dialog_indices.append({
                'start': len(utterances),
                'end': len(utterances) + len(dialog),
            })

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses

    digitpat = re.compile("\d+")
    timepat = re.compile("\d{1,2}[:]\d{1,2}")
    pricepat = re.compile("\d{1,3}[.]\d{1,2}")
    nested_brackets_pat = re.compile(r'\[([^\[\]]*)\[([^\|]*)\|[^\]]*\]')

    replacements = [(' one ', ' 1 '),
                    (' two ', ' 2 '),
                    (' three ', ' 3 '),
                    (' four ', ' 4 '),
                    (' five ', ' 5 '),
                    (' six ', ' 6 '),
                    (' seven ', ' 7 '),
                    (' eight ', ' 8 '),
                    (' nine ', ' 9 '),
                    (' ten ', ' 10 '),
                    (' eleven ', ' 11 '),
                    (' twelve ', ' 12 '),
                    (' thirteen ', ' 13 ')]

    @classmethod
    def sent_normalize(cls, text: str) -> str:

        def insertSpace(token, text):
            sidx = 0
            while True:
                sidx = text.find(token, sidx)
                if sidx == -1:
                    break
                if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                        re.match('[0-9]', text[sidx + 1]):
                    sidx += 1
                    continue
                if text[sidx - 1] != ' ':
                    text = text[:sidx] + ' ' + text[sidx:]
                    sidx += 1
                if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                    text = text[:sidx + 1] + ' ' + text[sidx + 1:]
                sidx += 1
            return text

        # lower case every word
        text = text.lower()

        # replace white spaces in front and end
        text = re.sub(r'^\s*|\s*$', '', text)

        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)

        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall("([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}"
                        "[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})", text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

        # weird unicode bug
        text = re.sub(u"(\u2018|\u2019)", "'", text)

        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')

        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\"|\<>@\(\)]', '', text)

        # insert white space before and after tokens:
        for token in ['?', '.', ',', '!', ':']:
            text = insertSpace(token, text)

        for fromx, tox in cls.replacements:
            text = ' ' + text + ' '
            text = text.replace(fromx, tox)[1:-1]

        # remove multiple spaces
        text = re.sub(' +', ' ', text)

        # concatenate numbers
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)

        return text

    @classmethod
    def delexicalise(cls, text: str, dictionary: Dict) -> str:
        # replace time and price
        text = re.sub(cls.timepat, ' [\g<0>|value_time] ', text)
        text = re.sub(cls.pricepat, ' [\g<0>|value_price] ', text)
        # replace all key-value pairs from databases
        for key, val in dictionary:
            text = (' ' + text + ' ').replace(' ' + key + ' ', ' ' + val + ' ')[1:-1]
        return text

    @classmethod
    def delexicalise_ref(cls, text: str, turn: Dict, domains: List) -> str:
        """
        Based on the belief state, we can find reference number that
        during data gathering was created randomly.
        """
        if turn['metadata']:
            for domain in domains:
                if domain not in turn['metadata']:
                    continue
                booked = turn['metadata'][domain]['book']['booked']
                if booked:
                    for slot in booked[0]:
                        key = cls.sent_normalize(booked[0][slot])
                        if slot == 'reference':
                            val = '[' + key + '|' + domain + '_' + slot + ']'
                        else:
                            val = '[' + key + '|' + domain + '_' + slot + ']'
                        text = (' ' + text + ' ').replace(' ' + key + ' ', ' '+val+' ')

                        # try reference with hashtag
                        key = cls.sent_normalize("#" + booked[0][slot])
                        text = (' ' + text + ' ').replace(' ' + key + ' ', ' '+val+' ')

                        # try reference with ref#
                        key = cls.sent_normalize("ref#" + booked[0][slot])
                        text = (' ' + text + ' ').replace(' ' + key + ' ', ' '+val+' ')
        return text

    @classmethod
    def remove_nested_slots(cls, text: str) -> str:
        for i in range(text.count('[')):
            text = cls.nested_brackets_pat.sub(r'[\1\2', text)
        return text

    @staticmethod
    def fix_delexicalization(filename, data, act_data, idx, idx_acts):
        """Given system dialogue acts fix automatic delexicalization."""
        fname = filename.strip('.json')
        if (fname not in act_data) or (str(idx_acts) not in act_data[fname]):
            return data
        turn = act_data[fname][str(idx_acts)]

        if not isinstance(turn, str):
            for k, act in turn.items():
                text = data['log'][idx]['text']
                if 'Attraction' in k:
                    if 'restaurant_' in text:
                        text = text.replace("restaurant_", "attraction_")
                    if 'hotel_' in text:
                        text = text.replace("hotel_", "attraction_")
                if 'Hotel' in k:
                    if 'attraction_' in text:
                        text = text.replace("attraction_", "hotel_")
                    if 'restaurant_' in text:
                        text = text.replace("restaurant_", "hotel_")
                if 'Restaurant' in k:
                    if 'attraction_' in text:
                        text = text.replace("attraction_", "restaurant_")
                    if 'hotel_' in text:
                        text = text.replace("hotel_", "restaurant_")
                data['log'][idx]['text'] = text

        return data

    @staticmethod
    def create_bio_markup(text: str) -> Tuple[List, List]:
        tokens, tags = [], []
        buff = ""
        slot_parse = False
        for ch in text:
            if ch == '[':
                slot_parse = True
            elif ch == ']':
                slot_parse = False
                slot_val, slot_type = buff.split('|')
                slot_val_toks = slot_val.split(' ')
                tokens.extend(slot_val_toks)
                tags.extend(['B-'+slot_type] + ['I-'+slot_type] * (len(slot_val_toks)-1))
                buff = ""
            elif not slot_parse and (ch == " "):
                if buff:
                    tokens.append(buff)
                    tags.append('O')
                    buff = ""
            else:
                buff += ch
        assert len(tokens) == len(tags), "nonmatching lengths of tokens and tags"
        return tokens, tags

    @classmethod
    def prepare_slot_dict(cls, filepaths: Dict[str, Union[str, Path]]) -> Dict:
        dic = []
        dic_train, dic_area, dic_food, dic_price = [], [], [], []

        def dic_append(val, raw_val, key, d=dic):
            d.append((val, '[' + raw_val + '|' + key + ']'))

        def uniquesOrdered(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        norm = cls.sent_normalize

        # read databases
        for domain, fname in filepaths.items():
            log.info(f"[parsing {fname}]")
            db_json = json.load(open(fname, 'rt'))

            for ent in db_json:
                if not isinstance(ent, dict):
                    break
                for key, val in ent.items():
                    if val == '?' or val == 'free':
                        continue

                    if isinstance(val, (int, list)):
                        val = str(int)
                    elif not isinstance(val, str):
                        print(f"pair `{key}, {val}` has nonstring type {type(val)}")
                        break
                    norm_val = norm(val)
                    if key == 'address':
                        dic_append(norm_val, norm_val, domain + '_address')
                        if "road" in val:
                            norm_val2 = norm(val.replace("road", "rd"))
                            dic_append(norm_val2, norm_val, domain + '_address')
                        elif "rd" in val:
                            norm_val2 = norm(val.replace("rd", "road"))
                            dic_append(norm_val2, norm_val, domain + '_address')
                        elif "st" in val:
                            norm_val2 = norm(val.replace("st", "street"))
                            dic_append(norm_val2, norm_val, domain + '_address')
                        elif "street" in val:
                            norm_val2 = norm(val.replace("street", "st"))
                            dic_append(norm_val2, norm_val, domain + '_address')
                    elif key == 'name':
                        dic_append(norm_val, norm_val, domain + '_name')
                        if "b & b" in val:
                            norm_val2 = norm(val.replace("b & b", "bed and breakfast"))
                            dic_append(norm_val2, norm_val, domain + '_name')
                        elif "bed and breakfast" in val:
                            norm_val2 = norm(val.replace("bed and breakfast", "b & b"))
                            dic_append(norm_val2, norm_val, domain + '_name')
                        elif "hotel" in val and 'gonville' not in val:
                            norm_val2 = norm(val.replace("hotel", ""))
                            dic_append(norm_val2, norm_val, domain + '_name')
                        elif "restaurant" in val:
                            norm_val2 = norm(val.replace("restaurant", ""))
                            dic_append(norm_val2, norm_val, domain + '_name')
                    elif key == 'postcode':
                        dic_append(norm_val, norm_val, domain + '_postcode')
                    elif key == 'phone':
                        dic_append(val, val, domain + '_phone')
                    elif key == 'trainID':
                        dic_append(norm_val, norm_val, domain + '_id')
                    elif key == 'department':
                        dic_append(norm_val, norm_val, domain + '_department')
                    elif key == 'departure' or key == 'destination':
                        dic_append(norm_val, norm_val, domain + '_place', d=dic_train)

                    # NORMAL DELEX
                    elif key == 'area':
                        dic_append(norm_val, norm_val, 'value_area', d=dic_area)
                    elif key == 'food':
                        dic_append(norm_val, norm_val, 'value_food', d=dic_food)
                    elif key == 'pricerange':
                        dic_append(norm_val, norm_val, 'value_pricerange', d=dic_price)
                    else:
                        pass

            if domain == 'hospital':
                for val, val_type in [('Hills Rd', 'address'), ('Hills Road', 'address'),
                                      ('CB20QQ', 'postcode'), ('CB11PS', 'postcode'),
                                      ('Addenbrookes Hospital', 'name')]:
                    norm_val = norm(val)
                    dic_append(norm_val, norm_val, domain + '_' + val_type)
                for val, val_type in [('01223245151', 'phone'), ('1223245151', 'phone'),
                                      ('0122324515', 'phone')]:
                    dic_append(val, val, domain + '_' + val_type)

            elif domain == 'police':
                for val, val_type in [('Parkside', 'address'), ('CB11JG', 'postcode'),
                                      ('Parkside Police Station', 'name')]:
                    norm_val = norm(val)
                    dic_append(norm_val, norm_val, domain + '_' + val_type)
                for val, val_type in [('01223358966', 'phone'), ('1223358966', 'phone')]:
                    dic_append(val, val, domain + '_' + val_type)

            elif domain == 'taxi':
                for color in db_json['taxi_colors']:
                    norm_val = norm(color)
                    dic_append(norm_val, norm_val, 'value' + '_color')
                for car_type in db_json['taxi_types']:
                    norm_val = norm(car_type)
                    dic_append(norm_val, norm_val, domain + '_type')

            elif domain == 'train':
                for val, val_type in [('Cambridge Towninfo Centre', 'place')]:
                    norm_val = norm(val)
                    dic_append(norm_val, norm_val, domain + '_' + val_type)

        # add specific values:
        for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
                    'sunday']:
            norm_key = norm(key)
            dic_append(norm_key, norm_key, 'value' + '_day')

        # more general values add at the end
        dic.extend(dic_train)
        dic.extend(dic_area)
        dic.extend(dic_food)
        dic.extend(dic_price)

        dic = uniquesOrdered(dic)
        dic = sorted(dic, key=len, reverse=True)
        log.info(f"[Dictionary size = {len(dic)}]")
        return dic

    @classmethod
    def preprocess_data(cls, data_file: Union[str, Path],
                        act_data_file: Union[str, Path],
                        databases: Dict[str, Union[str, Path]]) -> Dict:
        domains = list(databases.keys())
        db_dict = cls.prepare_slot_dict(databases)

        data = json.load(open(data_file, 'rt'))
        act_data = json.load(open(act_data_file, 'rt'))

        delex_data = {}
        for dialogue_name in tqdm(data):
            dialogue = data[dialogue_name]
            idx_acts = 1

            for idx, turn in enumerate(dialogue['log']):
                dialogue['log'][idx]['raw_text'] = turn['text']
                # normalization, split and delexicalization of the sentence
                sent = cls.sent_normalize(turn['text'])
                sent = cls.delexicalise(sent, db_dict)

                # parsing reference number GIDEN belief state
                sent = cls.delexicalise_ref(sent, turn, domains)

                # changes to number only here
                sent = re.sub(cls.digitpat, '[\g<0>|value_count]', sent)

                # remove nested entities
                sent = re.sub('\s+', ' ', cls.remove_nested_slots(sent)).strip()

                dialogue['log'][idx]['text'] = sent

                # FIXING delexicalization
                dialogue = cls.fix_delexicalization(dialogue_name, dialogue, act_data,
                                                     idx, idx_acts)

                # create bio-markup
                tokens, tags = cls.create_bio_markup(sent)
                dialogue['log'][idx]['tokens'] = tokens
                dialogue['log'][idx]['text'] = ' '.join(tokens)
                dialogue['log'][idx]['tags'] = tags

                idx_acts += 1

            delex_data[dialogue_name] = dialogue

        return delex_data
