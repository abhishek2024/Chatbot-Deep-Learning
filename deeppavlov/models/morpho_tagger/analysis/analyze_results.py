import sys
import os
import getopt
from collections import defaultdict
import ujson as json
import numpy as np

from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile


def indexes_to_tag(tags, indexes):
    answer = tags[indexes[0]]
    if len(indexes) > 0:
        for elem in indexes[1:]:
            answer += ";" + tags[elem].split("_")[1]
    return answer


def read_unimorph_file(infile):
    with open(infile, "r", encoding="utf8") as fin:
        data = json.load(fin)
    tags, indexes = data["pos"] + data["features"], data["word_indexes"]
    answer = dict()
    for word, word_indexes in indexes.items():
        word_indexes = [[int(x) for x in elem.split(",")] for elem in word_indexes.split(";")]
        answer[word] = [indexes_to_tag(tags, curr_indexes) for curr_indexes in word_indexes]
    return answer


if __name__ == "__main__":
    unimorph_file = None
    opts, args = getopt.getopt(sys.argv[1:], "u:")
    for opt, val in opts:
        if opt == "-u":
            unimorph_file = val
    results_file, corr_file, outfile, stats_outfile = args
    train_file = os.s
    pred_data, corr_data = read_infile(results_file), read_infile(corr_file)
    if unimorph_file is not None:
        unimorph_data = read_unimorph_file(unimorph_file)
    else:
        unimorph_data = None
    stats = defaultdict(lambda: defaultdict(int))
    with open(outfile, "w", encoding="utf8") as fout:
        for (sent, pred_tags), (_, corr_tags) in zip(pred_data, corr_data):
            for i, (word, tag, corr_tag) in enumerate(zip(sent, pred_tags, corr_tags), 1):
                stats[corr_tag][tag] += 1
                fout.write("{}\t{}\t{}\n".format(i, word, tag))
                if corr_tag != tag:
                    fout.write(corr_tag + "\n")
                if unimorph_data is not None and word in unimorph_data:
                    fout.write("Unimorph: " + "\t".join(unimorph_data[word]) + "\n")
            fout.write("\n")
    length_stats = {key: sum(value.values()) for key, value in stats.items()}
    error_stats = {key: length_stats[key] - stats[key][key] for key in length_stats}
    with open(stats_outfile, "w", encoding="utf8") as fout:
        for tag, tag_data in sorted(stats.items(), key=(lambda x: error_stats[x[0]]), reverse=True):
            fout.write("{}\t{}\t{}\t{:.2f}\n".format(
                tag, length_stats[tag], error_stats[tag], 1.0 - error_stats[tag] / length_stats[tag]))
            for other, count in sorted(tag_data.items(), key=(lambda x: x[1]), reverse=True):
                fout.write("{}\t{}\n".format(other, count))
            fout.write("\n")


