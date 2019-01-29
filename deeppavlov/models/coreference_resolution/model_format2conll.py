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

import collections
import operator


def output_conll(input_file, predictions):
    """
    Gets the string with conll file, and write there new coreference clusters from predictions.

    Args:
        input_file: dict with conll string
        predictions: dict new clusters

    Returns: modified string with conll file

    """
    prediction_map = {}
    # input_file = input_file['conll_str']
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k, v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k, v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    new_conll = ''
    for line in input_file.split('\n'):
        if line.startswith("#begin"):
            new_conll += line + '\n'
            continue
        elif line.startswith("#end document"):
            new_conll += line
            continue
        else:
            row = line.split()
            if len(row) == 0:
                new_conll += '\n'
                continue

            glen = 0
            for l in row[:-1]:
                glen += len(l)
            glen += len(row[:-1])

            doc_key = 'bc' + '{}_{}'.format(row[0], row[1])
            start_map, end_map, word_map = prediction_map[doc_key]
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            word_index += 1

            line = line[:glen] + row[-1]
            new_conll += line + '\n'

    return new_conll
