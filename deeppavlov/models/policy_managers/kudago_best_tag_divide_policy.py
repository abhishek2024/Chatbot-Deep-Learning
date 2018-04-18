# !/usr/bin/env python3
# coding: utf-8

import argparse
import collections
import numpy as np
import json
from sklearn import cluster
from pprint import pprint


def jl_load(fpath):
    with open(fpath, 'rt') as f:
        for ln in f:
            yield json.loads(ln)


def get_tags_d(data):
    tags_d = collections.Counter()
    for sample in data:
        tags_d.update(sample['tags'])
    return tags_d


def onehot_encode(tags, all_tags):
    """
    tags: list of lists of str tags
    all_tags: list of str tags
    Returns: np.array (num_samples, num_tags)
    """
    num_samples, num_tags = len(tags), len(all_tags)
    onehoted = np.zeros((num_samples, num_tags))
    for i, s_tags in enumerate(tags):
        s_filtered_tags = set.intersection(set(s_tags), set(all_tags))
        for t in s_filtered_tags:
            onehoted[i, all_tags.index(t)] = 1
    return onehoted


def cooccurance_matrix(label_data):
    cmatrix = np.dot(label_data.transpose(),label_data)

    # Compute cooccurrence matrix in percentage
    # FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
    #      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804
    cmatrix_diagonal = np.diagonal(cmatrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cmatrix_percentage = np.nan_to_num(np.true_divide(cmatrix, cmatrix_diagonal[:, None]))
    
    return cmatrix, cmatrix_percentage


def cluster_cmatrix(cooc, tags_l):
    dist = 1 - cooc / cooc.max()
    model = cluster.AgglomerativeClustering(n_clusters=30, affinity="precomputed",
                                            linkage='average').fit(dist)
    new_order = np.argsort(model.labels_)
    ordered_cooc = cooc[new_order][:, new_order]
    
    new_labels = model.labels_[new_order]
    new_tags = np.array(tags_l)[new_order]

    clusters = []
    for l in set(new_labels):
        clusters.append(list(new_tags[new_labels == l]))
    return clusters


def rate_divide_by_tags(data_tags, tags):
    """
    data_tags: np.array (num_samples x num_tags)
    tags: np.array (num_tags x 1) or (num_tags)
    """
    num_samples, num_tags = data_tags.shape
    num_nonnull_samples = np.sum(np.sum(data_tags, axis=1) > 0)
    matching = np.sum(np.multiply(data_tags, tags), axis=1) > 0
    return np.sum(matching) / num_nonnull_samples


def divide_by_tags(data_tags, tags, exclude=False):
    """
    data_tags: np.array (num_samples x num_tags)
    tags: np.array (num_tags x 1) or (num_tags)
    """
    num_samples, num_tags = data_tags.shape
    matching = (np.sum(np.multiply(data_tags, tags), axis=1) > 0).astype(int)
    if exclude:
        matching = 1 - matching
    return np.multiply(data_tags, matching[:, np.newaxis])


def best_divide(data_onehots, tags_l, clusters):
    """
    data_onehots: np.array (num_samples, num_tags)
    tags_l: list of str tags
    clusters: list of list of str tags
    """
    divide_rates = []
    for cluster in clusters:
        cluster_onehot = onehot_encode([cluster], tags_l)
        divide_rates.append(rate_divide_by_tags(data_onehots, cluster_onehot))
    best_idx = np.argmin(np.fabs(0.5 - np.array(divide_rates)))
    return clusters[best_idx], divide_rates[best_idx]


def filter_sample(s):
    fields = ['id', 'slug', 'title', 'address', 'subway', 'timetable', 'phone',
              'site_url', 'foreign_url', 'comments_count', 'categories', 'tags']
    return {f: s.get(f) for f in fields}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples", help="Path to file with samples")
    parser.add_argument("-c", "--clusters", default=None,
                        help="Path to file with clusters of tags.")
    return parser.parse_args()


def main():
    args = parse_args()

    data = list(jl_load(args.samples))
    if args.clusters is not None:
        cl_q_l = list(jl_load(args.clusters))
        questions = [cl_q[1] for cl_q in cl_q_l]
        clusters = [cl_q[0] for cl_q in cl_q_l]
        tags_l = list(set([t for cl in clusters for t in cl]))
    else:
        questions = None
        clusters = None
        tags_d = dict(get_tags_d(data).most_common(100))
        tags_l = list(tags_d.keys())

    data_tags_l = [s['tags'] for s in data]
    data_onehots = onehot_encode(data_tags_l, tags_l)

    if clusters is None:
        cooc1, cooc2 = cooccurance_matrix(data_onehots)
        clusters = cluster_cmatrix(cooc2, tags_l)
        print("Found clusters:")
        pprint(clusters)
        clusters += [[tag] for tag in tags_l]
    
    ans = ''
    while ans != 'end':
        bst_cluster, bst_rate = best_divide(data_onehots, tags_l, clusters)
        if questions is not None:
            print("\n{}".format(questions[clusters.index(bst_cluster)]))
        else:
            print("\nDo you want something from cluster {} (rate={:.3f})?"
                  .format(bst_cluster, bst_rate))
        ans = input(">")
        while ans not in ("yes", "no", "end"):
            print("Please answer 'yes', 'no' or 'end'.")
            ans = input(">")
        if ans == 'yes':
            bst_cluster_onehot = onehot_encode([bst_cluster], tags_l)
            data_onehots = divide_by_tags(data_onehots, bst_cluster_onehot)
        elif ans == 'no':
            bst_cluster_onehot = onehot_encode([bst_cluster], tags_l)
            data_onehots = divide_by_tags(data_onehots, bst_cluster_onehot, exclude=True)
        else:
            break
        nonnull_tag_ids = np.where(np.sum(data_onehots, axis=0) > 0)[0]
        print("{} remained tags to choose from = {}"
              .format(len(nonnull_tag_ids), [tags_l[i] for i in nonnull_tag_ids]))
        nonnull_sample_ids = np.where(np.sum(data_onehots, axis=1) > 0)[0]
        print("{} remaining samples to choose from.".format(len(nonnull_sample_ids)))
    nonnull_sample_ids = np.where(np.sum(data_onehots, axis=1) > 0)[0]
    print("Random result: {}".format(filter_sample(data[nonnull_sample_ids[0]])))


if __name__ == "__main__":
    main()

