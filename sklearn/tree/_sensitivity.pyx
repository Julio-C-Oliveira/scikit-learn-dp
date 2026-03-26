cdef class SensitivityCalculator:
    cdef double compute(self, int n_node_samples) noexcept nogil:
        return 0.0

cdef class GiniSensitivity(SensitivityCalculator):
    cdef double compute(self, int n_node_samples) noexcept nogil:
        """Sensibilidade Gini: 1/n"""
        if n_node_samples <= 0:
            return 0.0
        return 1.0 / n_node_samples

cdef class MSESensitivity(SensitivityCalculator):
    def __cinit__(
        self, 
        double g_max, 
        double g_min
        ):
        cdef double amplitude = g_max - g_min
        self.sq_amplitude = amplitude * amplitude

    cdef double compute(self, int n_node_samples) noexcept nogil:
        """Sensibilidade MSE: (G_max - G_min)^2 / n"""
        if n_node_samples <= 0:
            return 0.0
        return self.sq_amplitude / n_node_samples