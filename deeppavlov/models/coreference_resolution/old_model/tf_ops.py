from pathlib import Path
import tensorflow as tf

kernel_path = Path(__file__).resolve().parent
# compile_coreference(kernel_path, operational_system=os)
coref_op_library = tf.load_op_library(str(kernel_path.joinpath("coref_kernels.so")))

tf.NotDifferentiable("Spans")
tf.NotDifferentiable("Antecedents")
tf.NotDifferentiable("ExtractMentions")
tf.NotDifferentiable("DistanceBins")

# C++ operations
spans = coref_op_library.spans
distance_bins = coref_op_library.distance_bins
extract_mentions = coref_op_library.extract_mentions
get_antecedents = coref_op_library.antecedents
