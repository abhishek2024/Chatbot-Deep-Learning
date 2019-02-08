import tensorflow as tf

coref_op_library = tf.load_op_library("./coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
