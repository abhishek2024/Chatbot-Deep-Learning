"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from abc import ABCMeta
from functools import wraps

from six import with_metaclass

ROOT_VARIABLE_SCOPE = tf.get_variable_scope() 


def _scope_wrap(func, scope_name):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        #print("Wrapping `{}` in `{}` variable scope"\
        #      .format(func.__name__, scope_name))
        #with tf.name_scope(graph):
        #if tf.contrib.framework.get_name_scope() == scope_name:
        #print("Current scope = `{}`".format(tf.get_variable_scope().name))
        #if func.__name__ in ('_get_train_op', 'load'):
            #print("Wrapping in global variable scope")
        #    with tf.variable_scope(ROOT_VARIABLE_SCOPE):
        #        return func(*args, **kwargs)
        #if tf.get_variable_scope().name.contains(scope_name):
        if tf.get_variable_scope().name == scope_name:
            #print("Not wrapping!")
        #if tf.get_variable_scope().name != "":
            return func(*args, **kwargs)
        else:
            #print("Wrapping!")
            with tf.variable_scope(scope_name):
                return func(*args, **kwargs)
    return _wrapped


def _graph_wrap(func, graph):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        #print("Wrapping `{}` in `{}` graph".format(func.__name__, graph))
        with graph.as_default():
            return func(*args, **kwargs)
    return _wrapped


class TfModelMeta(with_metaclass(type, ABCMeta)):

    def __call__(cls, *args, **kwargs):
        from .keras_model import KerasModel
        if issubclass(cls, KerasModel):
            import keras.backend as K
            K.clear_session()

        obj = cls.__new__(cls)
        #obj.graph = tf.variable_scope(obj.__class__.__name__ + '-' + str(id(obj)))
        #obj.graph = tf.name_scope(obj.__class__.__name__ + '-' + str(id(obj)))
        #graph_wrap = True
        graph_wrap = False
        obj.graph = None
        obj.scope_name = None
        if graph_wrap:
            obj.graph = tf.Graph()
        else:
            obj.scope_name = obj.__class__.__name__ + '-' + str(id(obj))
        for meth in dir(obj):
            if meth == '__class__':
                continue
            attr = getattr(obj, meth)
            if callable(attr):
                if graph_wrap:
                    setattr(obj, meth, _graph_wrap(attr, obj.graph))
                else:
                    setattr(obj, meth, _scope_wrap(attr, obj.scope_name))
        obj.__init__(*args, **kwargs)
        return obj
