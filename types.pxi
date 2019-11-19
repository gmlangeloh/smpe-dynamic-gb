'''
Imports basic types used almost everywhere.
'''

from libcpp cimport bool

from sage.numerical.backends.glpk_backend cimport GLPKBackend
from sage.numerical.mip cimport MixedIntegerLinearProgram

from sage.rings.polynomial.multi_polynomial_libsingular cimport MPolynomial_libsingular
from sage.rings.polynomial.multi_polynomial_libsingular cimport MPolynomialRing_libsingular

from sage.modules.vector_integer_dense cimport Vector_integer_dense
from sage.matrix.matrix_real_double_dense cimport Matrix_real_double_dense
from sage.modules.vector_real_double_dense cimport Vector_real_double_dense

from sage.interfaces.singular import singular

# memory management courtesy C
cdef extern from "stdlib.h":
  ctypedef unsigned int size_t
  void *malloc(size_t size)
  void free(void *ptr)
