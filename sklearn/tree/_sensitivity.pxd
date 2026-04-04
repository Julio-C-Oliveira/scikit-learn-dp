cdef class Sensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSplitSensitivity(Sensitivity):
    pass 

cdef class MSESplitSensitivity(Sensitivity):
    cdef double sq_amplitude

cdef class ClassCounterSensitivity(Sensitivity):
    pass

cdef class SumSensitivity(Sensitivity):
    cdef double amplitude