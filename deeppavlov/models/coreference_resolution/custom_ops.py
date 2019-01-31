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
from pathlib import Path
from typing import Union


def compile_coreference(path: Union[Path, str], operational_system: str = "linux", mode: str = "pip") -> None:
    if isinstance(path, str):
        path = Path(path)

    if not path.joinpath('coref_kernels.so').exists():
        print('[ Compiling the custom tensorflow operations from coref_kernels.cc ]')

        if operational_system == "linux":
            if mode == "pip":
                # new
                gcc = "g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0"
                # old
                # gcc = "g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2"
            else:
                gcc = "g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2"
        elif operational_system == "mac":
            gcc = "g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup"
        elif operational_system == "windows":
            raise NotImplementedError("Windows OS support not implemented yet.")
        else:
            raise AttributeError(f"OS with name {operational_system} isn't supported.")

        cmd = f"""
               #!/usr/bin/env bash

               # Build custom kernels.
               TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
               TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

               {gcc}
               """

        os.system(cmd)
        print('[ Compilation complete. ]')
