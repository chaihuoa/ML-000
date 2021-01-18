from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

compile_flags = ['-std=c++11']
# linker_flags = ['-fopenmp']

module = Extension('target_encoding_v1_cy',
                   ['target_encoding_v1_cy.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()], # This helps to create numpy
                   extra_compile_args=compile_flags)

setup(
    name='target_encoding_v1_cy',
    ext_modules=cythonize(module)
)