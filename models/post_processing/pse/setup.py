from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    Extension('pse',
              sources=['pse.pyx'],
              language='c++',
              include_dirs=[numpy.get_include()],
              library_dirs=[],
              libraries=[],
              extra_compile_args=['-O3'],
              extra_link_args=[])))
