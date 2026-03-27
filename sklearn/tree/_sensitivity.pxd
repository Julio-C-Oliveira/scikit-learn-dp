cdef class Sensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSensitivity(Sensitivity):
    pass 

cdef class MSESensitivity(Sensitivity):
    cdef double sq_amplitude