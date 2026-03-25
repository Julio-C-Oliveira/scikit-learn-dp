# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _utils.pyx for details.

cimport numpy as cnp
from sklearn.tree._tree cimport Node
from sklearn.neighbors._quad_tree cimport Cell
from sklearn.utils._typedefs cimport float32_t, float64_t, intp_t, uint8_t, int32_t, uint32_t

# Modificado: Importa a eestrutura necessária.
from sklearn.tree._splitter cimport SplitRecordForDifferentialPrivacy


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    # It corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef float32_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (float32_t*)
    (intp_t*)
    (uint8_t*)
    (float64_t*)
    (float64_t**)
    (Node*)
    (Cell*)
    (Node**)

cdef int safe_realloc(realloc_ptr* p, size_t nelems) except -1 nogil


cdef cnp.ndarray sizet_ptr_to_ndarray(intp_t* data, intp_t size)


cdef intp_t rand_int(intp_t low, intp_t high,
                     uint32_t* random_state) noexcept nogil


cdef float64_t rand_uniform(float64_t low, float64_t high,
                            uint32_t* random_state) noexcept nogil


cdef float64_t log(float64_t x) noexcept nogil


cdef class WeightedFenwickTree:
    cdef intp_t size         # number of leaves (ranks)
    cdef float64_t* tree_w   # BIT for weights
    cdef float64_t* tree_wy  # BIT for weighted targets
    cdef intp_t max_pow2     # highest power of two <= n
    cdef float64_t total_w   # running total weight
    cdef float64_t total_wy  # running total weighted target

    cdef void reset(self, intp_t size) noexcept nogil
    cdef void add(self, intp_t idx, float64_t y, float64_t w) noexcept nogil
    cdef intp_t search(
        self,
        float64_t t,
        float64_t* cw_out,
        float64_t* cwy_out,
        intp_t* prev_idx_out,
    ) noexcept nogil

# =============================================================================
# DPNodeSplit for Differential Privacy data structure
# =============================================================================

cdef struct SplitRecordArray:
    SplitRecordForDifferentialPrivacy* data
    size_t size
    size_t capacity

cdef void init_array(SplitRecordArray* arr) noexcept nogil
cdef void free_array(SplitRecordArray* arr) noexcept nogil

cdef void append_to_array(SplitRecordArray* arr, SplitRecordForDifferentialPrivacy* value) noexcept nogil

cdef float64_t get_max_improvement_array(SplitRecordArray* arr) noexcept nogil
cdef float64_t get_min_improvement_array(SplitRecordArray* arr) noexcept nogil
cdef void downward_scaling_array(SplitRecordArray* arr, float64_t max_improvement) noexcept nogil

cdef void calculate_weights_and_probabilities(SplitRecordArray* arr, float64_t epsilon, float64_t delta_q) noexcept nogil

cdef float64_t random_float() noexcept nogil
cdef SplitRecordForDifferentialPrivacy* choose_weighted_random(SplitRecordArray* arr) noexcept nogil