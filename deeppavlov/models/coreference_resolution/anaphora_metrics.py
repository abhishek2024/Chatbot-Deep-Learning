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

import os
import uuid

import numpy as np
from tqdm import tqdm


def extract_data(infile):
    """
    Extract useful information from conll file and write it in special dict structure.

    {'doc_name': name,
     'parts': {'0': {'text': text,
                    'chains': {'1421': {'chain_id': 'bfdb6d6f-6529-46af-bd39-cf65b96479b5',
                                        'mentions': [{'sentence_id': 0,
                                                    'start_id': 0,
                                                    'end_id': 4,
                                                    'start_line_id': 1,
                                                    'end_line_id': 5,
                                                    'mention': 'Отличный отель для бюджетного отдыха',
                                                    'POS': ['Afpmsnf', 'Ncmsnn', 'Sp-g', 'Afpmsgf', 'Ncmsgn'],
                                                    'mention_id': 'f617f5ca-ed93-49aa-8eae-21b672c5a9df',
                                                    'mention_index': 0}
                                        , ... }}}}}

    Args:
        infile: path to ground conll file

    Returns:
        dict with an alternative mention structure
    """
    # filename, parts: [{text: [sentence, ...], chains}, ...]
    data = {'doc_name': None, 'parts': dict()}
    with open(infile, 'r', encoding='utf8') as fin:
        current_sentence = []
        current_sentence_pos = []
        sentence_id = 0
        opened_labels = dict()
        # chain: chain_id -> list of mentions
        # mention: sent_id, start_id, end_id, mention_text, mention_pos
        chains = dict()
        part_id = None
        mention_index = 0
        for line_id, line in enumerate(fin.readlines()):
            # if end document part then add chains to data
            if '#end document' == line.strip():
                assert part_id is not None, print('line_id: ', line_id)
                data['parts'][part_id]['chains'] = chains
                chains = dict()
                sentence_id = 0
                mention_index = 0
            # start or end of part
            if '#' == line[0]:
                continue
            line = line.strip()
            # empty line between sentences
            if len(line) == 0:
                assert len(current_sentence) != 0
                assert part_id is not None
                if part_id not in data['parts']:
                    data['parts'][part_id] = {'text': [], 'chains': dict()}
                data['parts'][part_id]['text'].append(' '.join(current_sentence))
                current_sentence = []
                current_sentence_pos = []
                sentence_id += 1
            else:
                # line with data
                line = line.split()
                assert len(line) >= 3, print('assertion:', line_id, len(line), line)
                current_sentence.append(line[3])
                doc_name, part_id, word_id, word, ref = line[0], int(line[1]), int(line[2]), line[3], line[-1]
                pos, = line[4],
                current_sentence_pos.append(pos)

                if data['doc_name'] is None:
                    data['doc_name'] = doc_name

                if ref != '-':
                    # we have labels like (322|(32)
                    ref = ref.split('|')
                    # get all opened references and set them to current word
                    for i in range(len(ref)):
                        ref_id = int(ref[i].strip('()'))
                        if ref[i][0] == '(':
                            if ref_id in opened_labels:
                                opened_labels[ref_id].append((word_id, line_id))
                            else:
                                opened_labels[ref_id] = [(word_id, line_id)]

                        if ref[i][-1] == ')':
                            assert ref_id in opened_labels
                            start_id, start_line_id = opened_labels[ref_id].pop()

                            if ref_id not in chains:
                                chains[ref_id] = {
                                    'chain_id': str(uuid.uuid4()),
                                    'mentions': [],
                                }

                            chains[ref_id]['mentions'].append({
                                'sentence_id': sentence_id,
                                'start_id': start_id,
                                'end_id': word_id,
                                'start_line_id': start_line_id,
                                'end_line_id': line_id,
                                'mention': ' '.join(current_sentence[start_id:word_id + 1]),
                                'POS': current_sentence_pos[start_id:word_id + 1],
                                'mention_id': str(uuid.uuid4()),
                                'mention_index': mention_index,
                            })
                            assert len(current_sentence[start_id:word_id + 1]) > 0, (doc_name, sentence_id, start_id,
                                                                                     word_id, current_sentence)
                            mention_index += 1
    return data


def ana_doc_score(g_file, p_file):
    """
    The function takes two files at the input, analyzes them and calculates anaphora score. A weak criterion
    for antecedent identification is used.
    In the document, all the pronouns are used.

    Args:
        g_file: path to ground truth conll file
        p_file: path to predicted conll file

    Returns:
        precision, recall, F1-metrics
    """

    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    score = 0
    fn = 0
    fp = 0

    for x in main:
        k = 0
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    score += 1
                    k += 1
                    continue
            if k == 0:
                if main[x]['g_clus'] == main[x]['p_clus']:
                    fn += 1
                    # print('fn', fn)
                    # print(main[x])
                else:
                    fp += 1
                    # print('fp', fp)
                    # print(main[x])
        else:
            continue

    precision = score / (score + fp)
    recall = score / (score + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def true_ana_score(g_file, p_file):
    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    tp = 0
    fn = 0
    fp = 0
    prec = list()
    rec = list()
    f_1 = list()

    for x in main:
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    tp += 1
                else:
                    fp += 1
            for y in main[x]['gold']:
                if y not in main[x]['pred']:
                    fn += 1

        if (tp + fp) == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        if (tp + fn) == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        if (p + r) != 0:
            f = 2 * p * r / (p + r)
        else:
            f = 0

        prec.append(p)
        rec.append(r)
        f_1.append(f)

    precision = np.mean(prec)
    recall = np.mean(rec)
    f1 = np.mean(f_1)

    return precision, recall, f1


def pos_ana_score(g_file, p_file):
    """
    Anaphora scores from gold and predicted files.
    A weak criterion for antecedent identification is used. In documents only used 3d-person pronouns,
    reflexive pronouns and relative pronouns.

    Args:
        g_file: path to ground truth conll file
        p_file: path to predicted conll file

    Returns:
        precision, recall, F1-metrics
    """

    main = {}
    reflexive_p = ['себя', 'себе', 'собой', 'собою', 'свой', 'своя', 'своё', 'свое', 'свои',
                   'своего', 'своей', 'своего', 'своих', 'своему', 'своей', 'своему',
                   'своим', 'своего', 'свою', 'своего', 'своих', 'своим', 'своей', 'своим',
                   'своими', 'своём', 'своем', 'своей', 'своём', 'своем', 'своих']
    relative_p = ['кто', 'что', 'какой', 'каков', 'который', 'чей', 'сколько', 'где', 'куда', 'когда',
                  'откуда', 'почему', 'зачем', 'как']

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    # else:
    #     print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id']:
                    if z['POS'][0].startswith('P-3'):
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

                    elif z['mention'] in relative_p:
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

                    elif z['mention'] in reflexive_p:
                        c = (z['start_line_id'], z['end_line_id'])
                        if (z['start_line_id'], z['end_line_id']) not in main:
                            main[c] = dict()
                            main[c]['pred'] = list()
                            main[c]['gold'] = list()
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y
                        else:
                            if x is g_data:
                                main[c]['g_clus'] = gnames.index(y)
                                main[c]['g_clus_real'] = y
                            else:
                                main[c]['p_clus'] = pnames.index(y)
                                main[c]['p_clus_real'] = y

    for x in main:
        try:
            for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In gold file {0} absent cluster {1}".format(g_file, x))
            continue

        try:
            for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
                if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
                elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                    main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
        except KeyError:
            print("In predicted file {0} absent cluster {1}".format(p_file, x))
            continue

    # compute F1 score
    tp = 0
    fn = 0
    fp = 0
    prec = list()
    rec = list()
    f_1 = list()

    for x in main:
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    tp += 1
                else:
                    fp += 1
            for y in main[x]['gold']:
                if y not in main[x]['pred']:
                    fn += 1

        if (tp + fp) == 0:
            p = 0
        else:
            p = tp / (tp + fp)

        if (tp + fn) == 0:
            r = 0
        else:
            r = tp / (tp + fn)

        if (p + r) == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        prec.append(p)
        rec.append(r)
        f_1.append(f)

    precision = np.mean(prec)
    recall = np.mean(rec)
    f1 = np.mean(f_1)

    return precision, recall, f1


def anaphora_score(keys_path, predicts_path, stype):
    """
    Anaphora scores predicted files.
    A weak criterion for antecedent identification is used.
    The function takes the input to the path to the folder containing the key files, and the path to the folder
    containing the predicted files and the scorer type. It calculates the anaphora score for each file in folders,
    and averages the results. The number of files in the packs must be the same, otherwise the function
    will generate an error.

    Args:
        keys_path: path to ground truth conll files
        predicts_path: path to predicted conll files
        stype: a trigger that selects the type of scorer to be used

    Returns:
        dict with scores
    """
    pred_files = list(filter(lambda x: x.endswith('conll'), os.listdir(predicts_path)))

    # assert len(key_files) == len(pred_files), ('The number of visible files is not equal.')

    results = dict()
    results['precision'] = list()
    results['recall'] = list()
    results['F1'] = list()

    result = dict()

    for file in tqdm(pred_files):
        predict_file = os.path.join(predicts_path, file)
        gold_file = os.path.join(keys_path, file)

        if stype == 'full':
            p, r, f1 = true_ana_score(gold_file, predict_file)
        elif stype == 'semi':
            p, r, f1 = ana_doc_score(gold_file, predict_file)
        elif stype == 'pos':
            p, r, f1 = pos_ana_score(gold_file, predict_file)
        else:
            raise ValueError('Non supported type of anaphora scorer. {}'.format(stype))

        results['precision'].append(p)
        results['recall'].append(r)
        results['F1'].append(f1)

    result['precision'] = np.mean(results['precision'])
    result['recall'] = np.mean(results['recall'])
    result['F1'] = np.mean(results['F1'])

    return result
