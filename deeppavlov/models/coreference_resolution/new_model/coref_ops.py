from pathlib import Path

import tensorflow as tf

kernel_path = Path(__file__).resolve().parent
coref_op_library = tf.load_op_library(str(kernel_path.joinpath("coref_kernels.so")))

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
