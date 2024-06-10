import numpy
import os
import subprocess
from distutils.core import setup, Extension
from Cython.Build import cythonize
from pathlib import Path
from typing import List


class LPCNetExtension(Extension):
    """
    Extended extension class that runs the autogen.h and configure script before installing the LPCNet wrappers.
    """
    def __init__(self, name, sources, *args, **kw):
        self.lpcnet_submodule_dir = "LPCNet"
        self.autogen_script = os.path.join(self.lpcnet_submodule_dir, 'autogen.sh')
        self.configure_script = os.path.join(self.lpcnet_submodule_dir, 'configure')

        self._run_autogen_and_configure()
        super().__init__(name, sources, *args, **kw)

    def _run_autogen_and_configure(self):
        # call ./autogen.sh
        subprocess.call(['bash', './autogen.sh'], cwd=self.lpcnet_submodule_dir)

        # call ./configure
        cflags = os.environ.copy()
        cflags['CFLAGS'] = '-Ofast -g -march=native'
        subprocess.call(['bash', './configure'], cwd=self.lpcnet_submodule_dir, env=cflags)

    @staticmethod
    def get_c_source_files() -> List[str]:
        base_path = Path("LPCNet/src")
        c_source_files = ["ceps_codebooks.c", "kiss99.c", "lpcnet_enc.c", "ceps_vq_train.c", "kiss_fft.c",
                          "lpcnet_plc.c", "common.c", "lpcnet.c", "nnet.c", "lpcnet_dec.c", "nnet_data.c", "freq.c",
                          "pitch.c"]

        return [(base_path / c_src).as_posix() for c_src in c_source_files]

    @staticmethod
    def get_c_header_files() -> List[str]:
        return [".", "LPCNet/include", "LPCNet/src"]


setup(
    setup_requires=['numpy'],
    ext_modules=cythonize([
        LPCNetExtension("LPCNet",
                        sources=["LPCNet.pyx", *LPCNetExtension.get_c_source_files()],
                        include_dirs=[numpy.get_include(), *LPCNetExtension.get_c_header_files()],
                        language="c",
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ], compiler_directives={"language_level": "3"}),
    zip_safe=False
)
