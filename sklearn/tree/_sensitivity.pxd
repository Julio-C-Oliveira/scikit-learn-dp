cdef class SensitivityCalculator:
    cdef double compute(self, int n_node_samples) noexcept nogil

cdef class GiniSensitivity(SensitivityCalculator):
    pass 

cdef class MSESensitivity(SensitivityCalculator):
    cdef double sq_amplitude