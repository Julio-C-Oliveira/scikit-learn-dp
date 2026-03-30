cdef class SplitSensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSplitSensitivity(SplitSensitivity):
    pass 

cdef class MSESplitSensitivity(SplitSensitivity):
    cdef double sq_amplitude

cdef class ClassCounterSensitivity(SplitSensitivity):
    pass

cdef class SumSensitivity(SplitSensitivity):
    cdef double amplitude