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

import pickle
import fastText
import nltk
from ufal_udpipe import Model as udModel, Pipeline
from udapi.block.read.conllu import Conllu
from io import StringIO
from pathlib import Path
from string import punctuation
from logging import getLogger
from typing import List, Tuple, Optional, Dict

import numpy as np

from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.kbqa.entity_linking import EntityLinker

log = getLogger(__name__)

def descendents(node, desc_list):
    if len(node.children) > 0:
        for child in node.children:
            desc_list = descendents(child, desc_list)
    desc_list.append(node.form)

    return desc_list


@register('kb_rule')
class KBRule(Component, Serializable):
    """
        This class generates an answer for a given question using Wikidata.
        It searches for matching triplet from the Wikidata with entity and
        relation mentioned in the question. It uses results of the Named
        Entity Recognition component to extract entity mention and Classification
        component to determine relation which connects extracted entity and the
        answer entity.
    """

    def __init__(self, load_path: str, linker: EntityLinker, debug: bool = False,
                 relations_maping_filename: str = None, fasttext_load_path: str = None,
                 templates_filename: str = None, udpipe_path: str = None, *args, **kwargs) -> None:
        
        super().__init__(save_path=None, load_path=load_path)
        self._debug = debug
        self._relations_filename = relations_maping_filename
        self.q_to_name: Optional[Dict[str, Dict[str, str]]] = None
        self._relations_mapping: Optional[Dict[str, str]] = None
        self._templates_filename = templates_filename
        self.linker = linker
        self.fasttext = fastText.load_model(str(fasttext_load_path))
        self.ud_model = udModel.load(udpipe_path)
        self.full_ud_model = Pipeline(self.ud_model, "vertical", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        self.load()

    def load(self) -> None:
        with open(self.load_path, 'rb') as fl:
            self.q_to_name = pickle.load(fl)
        if self._relations_filename is not None:
            with open(self.load_path.parent / self._relations_filename, 'rb') as f:
                self._relations_mapping = pickle.load(f)
        if self._templates_filename is not None:
            with open(self.load_path.parent / self._templates_filename, 'rb') as t:
                self.templates = pickle.load(t)

    def save(self) -> None:
        pass

    def __call__(self, sentences: List[str],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        for sentence in sentences:
            is_kbqa = self.is_kbqa_question(sentence)
            if is_kbqa:
                obj = ""
                q_tokens = nltk.word_tokenize(sentence)
                entity_from_template, relation_from_template = self.entities_and_rels_from_templates(q_tokens)
                if entity_from_template:
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_template, q_tokens)
                    obj = self.match_triplet(entity_triplets, [relation_from_template])
                    objects_batch.append(obj)
                else:
                    detected_entity, detected_rel = self.entities_and_rels_from_tree(q_tokens)
                    if detected_entity:
                        print("detected_entity", detected_entity, "detected_rel", detected_rel)
                        entity_triplets, entity_linking_confidences = self.linker(detected_entity, q_tokens)
                        entity_triplets_flat = [item for sublist in entity_triplets for item in sublist]
                        obj = self.match_rel(entity_triplets_flat, detected_rel)
                        objects_batch.append(obj)
                    else:
                        objects_batch.append('')
            else:
                objects_batch.append('')

        parsed_objects_batch = self.parse_wikidata_object(objects_batch)

        return parsed_objects_batch
        
    def find_entity(self, tree, q_tokens):
        detected_entity = ""
        detected_rel = ""
        min_tree = 10
        leaf_node = None
        for node in tree.descendants:
            if len(node.children) < min_tree and node.upos in ["NOUN", "PROPN"]:
                leaf_node = node

        if leaf_node is not None:
            node = leaf_node
            desc_list = []
            entity_tokens = []
            while node.parent.upos in ["NOUN", "PROPN"] and node.parent.deprel!="root" and not node.parent.parent.form.startswith("Как"):
                node = node.parent
            detected_rel = node.parent.form
            desc_list.append(node.form)
            desc_list = descendents(node, desc_list)
            num_tok = 0
            for n, tok in enumerate(q_tokens):
                if tok in desc_list:
                    entity_tokens.append(tok)
                    num_tok = n
            if (num_tok+1) < len(q_tokens):
                if q_tokens[(num_tok+1)].isdigit():
                    entity_tokens.append(q_tokens[(num_tok+1)])
            detected_entity = ' '.join(entity_tokens)
            return True, detected_entity, detected_rel

        return False, detected_entity, detected_rel

    def find_entity_adj(self, tree):
        detected_rel = ""
        detected_entity = ""
        for node in tree.descendants:
            if len(node.children) <= 1 and node.upos == "ADJ":
                detected_rel = node.parent.form
                detected_entity = node.form
                return True, detected_entity, detected_rel
        
        return False, detected_entity, detected_rel

    def match_rel(self, entity_triplets, detected_rel):
        av_detected_emb = self.av_emb(detected_rel)
        max_score = 0.0
        wiki_rel = ""
        found_obj = ""
        for triplet in entity_triplets:
            scores = []
            rel_id = triplet[0]
            obj = triplet[1]
            if rel_id in self._relations_mapping:
                rel_name = self._relations_mapping[rel_id]["name"]
                if rel_name == detected_rel:
                    wiki_rel = rel_name
                    found_obj = obj
                    return found_obj
                else:
                    name_emb = self.av_emb(rel_name)
                    scores.append(np.dot(av_detected_emb, name_emb))
                    if "aliases" in self._relations_mapping[rel_id]:
                        rel_aliases = self._relations_mapping[rel_id]["aliases"]
                        for alias in rel_aliases:
                            if alias == detected_rel:
                                wiki_rel = alias
                                found_obj = obj
                                return found_obj
                            else:
                                alias_emb = self.av_emb(alias)
                                scores.append(np.dot(av_detected_emb, alias_emb))
                    if np.asarray(scores).mean() > max_score:
                        max_score = np.asarray(scores).mean()
                        wiki_rel = rel_name
                        found_obj = obj
                    print(rel_name, rel_id, np.asarray(scores).mean())
        print("wiki_rel", wiki_rel, max_score)

        return found_obj

    def av_emb(self, rel):
        rel_tokens = nltk.word_tokenize(rel)
        emb = []
        for tok in rel_tokens:
            emb.append(self.fasttext.get_word_vector(tok))
        av_emb = np.asarray(emb).mean(axis = 0)
        return av_emb

    def is_kbqa_question(self, question_init):
        not_kbqa_question_templates = ["почему", "когда будет", "что будет", "что если", "для чего ", "как ", \
                                       "что делать", "зачем", "что может"]
        kbqa_question_templates = ["как зовут", "как называется", "как звали", "как ты думаешь", "как твое мнение", \
                                   "как ты считаешь"]

        question = ''.join([ch for ch in question_init if ch not in punctuation]).lower()
        print("question", question)
        is_kbqa = (all(template not in question for template in not_kbqa_question_templates) or
                   any(template in question for template in kbqa_question_templates))
        print("is_kbqa", is_kbqa)
        return is_kbqa

    def parse_wikidata_object(self, objects_batch):
        parsed_objects = []
        for n, obj in enumerate(objects_batch):
            if len(obj) > 0:
                if obj.startswith('Q'):
                    if obj in self.q_to_name:
                        parsed_object = self.q_to_name[obj]["name"]
                        parsed_objects.append(parsed_object)
                    else:
                        parsed_objects.append('Not Found')
                else:
                    parsed_objects.append(obj)
            else:
                parsed_objects.append('Not Found')
        return parsed_objects

    def entities_and_rels_from_templates(self, tokens):
        s_sanitized = ' '.join([ch for ch in tokens if ch not in punctuation]).lower()
        ent = ''
        relation = ''
        for template in self.templates:
            template_start, template_end = template.lower().split('xxx')
            if template_start in s_sanitized and template_end in s_sanitized:
                template_start_pos = s_sanitized.find(template_start)
                template_end_pos = s_sanitized.find(template_end)
                ent_cand = s_sanitized[template_start_pos+len(template_start): template_end_pos or len(s_sanitized)]
                if len(ent_cand) < len(ent) or len(ent) == 0:
                    ent = ent_cand
                    relation = self.templates[template]
        return ent, relation

    def entities_and_rels_from_tree(self, q_tokens):
        q_str = '\n'.join(q_tokens)
        s = self.full_ud_model.process(q_str)
        tree = Conllu(filehandle=StringIO(s)).read_tree()
        fnd = False
        fnd, detected_entity, detected_rel = self.find_entity(tree, q_tokens)
        if fnd == False:
            fnd, detected_entity, detected_rel = self.find_entity_adj(tree)
        detected_entity = detected_entity.replace("первый ", '')
        return detected_entity, detected_rel

    def match_triplet(self, entity_triplets, relations):
        obj = ''
        for predicted_relation in relations:
            for entities in entity_triplets:
                for rel_triplets in entities:
                    relation_from_wiki = rel_triplets[0]
                    if predicted_relation == relation_from_wiki:
                        obj = rel_triplets[1]
                        return obj
        return obj

