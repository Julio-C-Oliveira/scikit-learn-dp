cdef class Sensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSplitSensitivity(Sensitivity):
    pass 

cdef class MSESplitSensitivity(Sensitivity):
    cdef double sq_amplitude

cdef class ClassCounterSensitivity():
    cdef double compute(self) noexcept nogil

cdef class SumSensitivity(Sensitivity):
    cdef double g_max
    cdef double g_min
    cdef double amplitude