import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize


setup(
    setup_requires=['numpy'],
    ext_modules=cythonize([
        Extension(name="hga_optimized", sources=["hga_optimized.pyx", ], include_dirs=[numpy.get_include()],
                  define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
    ], compiler_directives={"language_level": "3"}),
    zip_safe=False
)
