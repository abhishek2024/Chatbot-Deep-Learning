import math
import random
from typing import Any

import h5py
import numpy as np
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.coreference_resolution.new_model import coref_ops, custom_layers


@register("coref_new_model")
class CorefModel(TFModel):
    """
    End-to-end neural model for coreference resolution.
    Class that create model from <paper>
    """

    def __init__(self,
                 context_embedder: Any = None,
                 head_embedder: Any = None,
                 char_dict: Any = None,
                 lm_path: str = "elmo_cache.hdf5",
                 use_metadata: bool = True,
                 use_features: bool = True,
                 model_heads: bool = True,
                 coarse_to_fine: bool = True,
                 filter_widths: list = (3, 4, 5),
                 filter_size: int = 50,
                 char_embedding_size: int = 8,
                 contextualization_size: int = 200,
                 contextualization_layers: int = 3,
                 ffnn_size: int = 150,
                 ffnn_depth: int = 2,
                 feature_size: int = 20,
                 max_span_width: int = 30,
                 coref_depth: int = 2,
                 lm_layers: int = 3,
                 lm_size: int = 1024,
                 random_seed: int = 42,
                 max_top_antecedents: int = 50,
                 max_training_sentences: int = 50,
                 top_span_ratio: float = 0.4,
                 max_gradient_norm: float = 5.0,
                 lstm_dropout_rate: float = 0.4,
                 lexical_dropout_rate: float = 0.5,
                 dropout_rate: float = 0.2,
                 optimizer: str = "adam",
                 learning_rate: float = 0.001,
                 decay_rate: float = 0.999,
                 decay_frequency: int = 100,
                 final_rate: float = 0.0002,
                 genres: list = ("bc", "bn", "mz", "nw", "pt", "tc", "wb"),
                 gpu_memory_fraction: float = 0.98,
                 **kwargs):
        # Parameters
        # ---------------------------------------------------------------------------------
        self.max_top_antecedents = max_top_antecedents
        self.max_training_sentences = max_training_sentences
        self.top_span_ratio = top_span_ratio

        # Model hyperparameters.
        self.filter_widths = filter_widths
        self.filter_size = filter_size
        self.char_embedding_size = char_embedding_size
        self.context_embeddings = context_embedder
        self.head_embeddings = head_embedder
        self.contextualization_size = contextualization_size
        self.contextualization_layers = contextualization_layers
        self.ffnn_size = ffnn_size
        self.ffnn_depth = ffnn_depth
        self.feature_size = feature_size
        self.max_span_width = max_span_width
        self.use_metadata = use_metadata
        self.use_features = use_features
        self.model_heads = model_heads
        self.coref_depth = coref_depth
        self.lm_layers = lm_layers
        self.lm_size = lm_size
        self.coarse_to_fine = coarse_to_fine

        # Learning hyperparameters.
        self.max_gradient_norm = max_gradient_norm
        self.lstm_dropout_rate = lstm_dropout_rate
        self.lexical_dropout_rate = lexical_dropout_rate
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_frequency = decay_frequency
        self.final_rate = final_rate

        # Other.
        self.char_dict = char_dict
        self.genres = genres
        self.random_seed = random_seed
        self.dropout = None
        self.lexical_dropout = None
        self.lstm_dropout = None
        self.lm_weights = None
        self.lm_scaling = None
        self.head_scores = None
        self.eval_data = None  # Load eval data lazily.

        self.genres = {g: i for i, g in enumerate(self.genres)}

        if lm_path:
            self.lm_file = h5py.File(lm_path, "r")
        else:
            raise ValueError("For the model to work, vectorized dataset is required. "
                             "Specify the value of the parameter 'lm_path'.")

        input_props = list()
        input_props.append((tf.float32, [None, None, self.context_embeddings.dim]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.dim]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(learning_rate, self.global_step,
                                                   decay_frequency, decay_rate,
                                                   staircase=True)
        # this is  training hack
        # learning_rate = tf.cond(learning_rate < self.final_rate,
        #                         lambda: tf.Variable(self.final_rate, tf.float32),
        #                         lambda: learning_rate)

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        optimizers = {"adam": tf.train.AdamOptimizer, "sgd": tf.train.GradientDescentOptimizer}
        optimizer = optimizers[optimizer](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

        tf.set_random_seed(self.random_seed)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)
        self.load()

    def start_enqueue_thread(self, train_example, is_training, returning=False):
        tensorized_example = self.tensorize_example(train_example, is_training=is_training)
        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
        self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        if returning:
            return tensorized_example

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        # file_key = doc_key.replace("/", ":")
        group = self.lm_file[doc_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    @staticmethod
    def tensorize_mentions(mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    @staticmethod
    def tensorize_span_labels(tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]
        gold_mentions = sorted(tuple(m) for m in custom_layers.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = custom_layers.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.filter_widths))
        text_len = np.array([len(s) for s in sentences])
        # word embeddings
        context_word_emb = self.context_embeddings(sentences)
        head_word_emb = self.head_embeddings(sentences)
        # char embeddings
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        lm_emb = self.load_lm_embeddings(doc_key)

        example_tensors = (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre,
                           is_training, gold_starts, gold_ends, cluster_ids)

        if is_training and len(sentences) > self.max_training_sentences:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors

    def truncate_example(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids,
                         genre, is_training, gold_starts, gold_ends, cluster_ids):
        max_training_sentences = self.max_training_sentences
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return (context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
                gold_starts, gold_ends, cluster_ids)

    @staticmethod
    def get_candidate_labels(candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    @staticmethod
    def get_dropout(dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = custom_layers.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                             0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = custom_layers.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = custom_layers.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = custom_layers.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    @staticmethod
    def distance_pruning(top_span_emb, top_span_mention_scores, c):
        k = custom_layers.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])  # [k, c]
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets  # [k, c]
        top_antecedents_mask = raw_top_antecedents >= 0  # [k, c]
        top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores,
                                                                                            top_antecedents)  # [k, c]
        top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_predictions_and_loss(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids,
                                 genre, is_training, gold_starts, gold_ends, cluster_ids):
        self.dropout = self.get_dropout(self.dropout_rate, is_training)
        self.lexical_dropout = self.get_dropout(self.lexical_dropout_rate, is_training)
        self.lstm_dropout = self.get_dropout(self.lstm_dropout_rate, is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.char_embedding_size > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.char_embedding_size]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb,
                                            [num_sentences * max_sentence_length, custom_layers.shape(char_emb, 2),
                                             custom_layers.shape(char_emb, 3)])
            # [num_sentences * max_sentence_length, max_word_length, emb]

            flattened_aggregated_char_emb = custom_layers.cnn(flattened_char_emb, self.filter_widths, self.filter_size)
            # [num_sentences * max_sentence_length, emb]

            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             custom_layers.shape(
                                                                                 flattened_aggregated_char_emb, 1)])
            # [num_sentences, max_sentence_length, emb]

            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        # if not self.lm_file:
        #     elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
        #     lm_embeddings = elmo_module(
        #         inputs={"tokens": tokens, "sequence_len": text_len},
        #         signature="tokens", as_dict=True)
        #     word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
        #     lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
        #                        lm_embeddings["lstm_outputs1"],
        #                        lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]

        lm_emb_size = custom_layers.shape(lm_emb, 2)
        lm_num_layers = custom_layers.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(
                tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1))
        # [num_sentences * max_sentence_length * emb, 1]

        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]

        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]
        num_words = custom_layers.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.feature_size]),
                              genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                   [1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))
        # [num_words, max_span_width]

        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))
        # [num_words, max_span_width]

        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]

        # candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
        #                                              flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts,
                                               candidate_ends)  # [num_candidates, emb]
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb)  # [k, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.top_span_ratio))
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   custom_layers.shape(context_outputs, 0),
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        # top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k]
        top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]

        c = tf.minimum(self.max_top_antecedents, k)

        if self.coarse_to_fine:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
                self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
                self.distance_pruning(top_span_emb, top_span_mention_scores, c)

        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.coref_depth):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                top_antecedent_scores = \
                    top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                 top_antecedents,
                                                                                 top_antecedent_emb,
                                                                                 top_antecedent_offsets,
                                                                                 top_span_speaker_ids,
                                                                                 genre_emb)  # [k, c]
                top_antecedent_weights = tf.nn.softmax(
                    tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1)
                # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                  1)  # [k, emb]
                with tf.variable_scope("f"):
                    f = tf.sigmoid(custom_layers.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                            custom_layers.shape(top_span_emb, -1)))  # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]
        loss = tf.reduce_sum(loss)  # []

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedents, top_antecedent_scores], loss

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.use_features:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.max_span_width, self.feature_size]),
                span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.model_heads:
            span_indices = tf.expand_dims(tf.range(self.max_span_width), 0) + tf.expand_dims(span_starts,
                                                                                             1)  # [k, max_span_width]
            span_indices = tf.minimum(custom_layers.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
            with tf.variable_scope("head_scores"):
                self.head_scores = custom_layers.projection(context_outputs, 1)  # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.max_span_width, dtype=tf.float32),
                                       2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return custom_layers.ffnn(span_emb, self.ffnn_depth, self.ffnn_size, 1,
                                      self.dropout)  # [k, 1]

    @staticmethod
    def softmax_loss(antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    @staticmethod
    def bucket_distance(distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb):
        k = custom_layers.shape(top_span_emb, 0)
        c = custom_layers.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.use_metadata:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.feature_size]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.use_features:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.feature_size]),
                antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = custom_layers.ffnn(pair_emb, self.ffnn_depth, self.ffnn_size,
                                                        1,
                                                        self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(
                custom_layers.projection(top_span_emb, custom_layers.shape(top_span_emb, -1)),
                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    @staticmethod
    def flatten_emb_by_sentence(emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, custom_layers.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.contextualization_layers):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = custom_layers.CustomLSTMCell(self.contextualization_size, num_sentences,
                                                           self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = custom_layers.CustomLSTMCell(self.contextualization_size, num_sentences,
                                                           self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                              cell_bw=cell_bw,
                                                                              inputs=current_inputs,
                                                                              sequence_length=text_len,
                                                                              initial_state_fw=state_fw,
                                                                              initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(custom_layers.projection(text_outputs, custom_layers.shape(text_outputs,
                                                                                                          2)))
                    # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    @staticmethod
    def get_predicted_antecedents(antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    @staticmethod
    def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def train_on_batch(self, *args):
        """
        Run train operation on one batch/document

        Args:
            args: (sentences, speakers, doc_key, clusters) list of text documents, list of authors, list of files names,
             list of true clusters

        Returns: Loss functions value and tf.global_step
        """
        sentences, speakers, doc_key, clusters = args
        batch = {"sentences": sentences[0], "speakers": speakers[0], "doc_key": doc_key[0], "clusters": clusters[0]}
        self.start_enqueue_thread(batch, True)
        tf_loss, tf_global_step, _ = self.sess.run([self.loss, self.global_step, self.train_op])
        return tf_loss

    def __call__(self, *args):
        sentences, speakers, doc_key = args
        batch = {"sentences": sentences[0], "speakers": speakers[0], "doc_key": doc_key[0], "clusters": []}
        self.start_enqueue_thread(batch, False)
        (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents,
         top_antecedent_scores) = self.sess.run(self.predictions)
        predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)

        return [predicted_clusters], [mention_to_predicted]

    def destroy(self):
        """Reset the model"""
        self.sess.close()
