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


def compile_coreference(path):
    path = path + 'coref_kernels.so'
    if not os.path.isfile(path):
        print('Compiling the coref_kernels.cc')
        cmd = """
              #!/usr/bin/env bash

              TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
              TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
              
              # Linux (pip)
              g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

              # Linux (build from source)
              #g++ -std=c++11 -shared ./coref_kernels.cc -o {0} -I $TF_INC -fPIC 

              # Mac
              #g++ -std=c++11 -shared ./coref_kernels.cc -o {0} -I $TF_INC -fPIC -D_GLIBCXX_USE_CXX11_ABI=0  -undefined dynamic_lookup
              """
        cmd = cmd.format(path)
        os.system(cmd)
        print('End of compiling the coref_kernels.cc')


if __name__ == "__main__":
    PATH = '.'
    compile_coreference(PATH)

# # Build custom kernels.
# TF_INC =$(python3 - c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#
# # Linux (pip)
# g + + -std = c + +11 - shared. / coref_kernels.cc - o
# {0} - I $TF_INC - fPIC - D_GLIBCXX_USE_CXX11_ABI = 0
