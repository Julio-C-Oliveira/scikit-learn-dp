cdef class Sensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSplitSensitivity(Sensitivity):
    pass 

cdef class MSESplitSensitivity(Sensitivity):
    cdef double sq_amplitude

cdef class LeafSensitivity:
    cdef double compute(self) noexcept nogil

cdef class ClassCounterSensitivity(LeafSensitivity):
    pass

cdef class SumSensitivity(LeafSensitivity):
    cdef double g_max
    cdef double g_min
    cdef double amplitude