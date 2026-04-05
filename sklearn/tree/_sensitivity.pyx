from libc.math cimport fabs, fmax

cdef class Sensitivity:
    cdef double compute(self, int n_node_samples) noexcept nogil:
        return 0.0

cdef class GiniSplitSensitivity(Sensitivity):
    cdef double compute(self, int n_node_samples) noexcept nogil:
        """Sensibilidade Gini: 1/n"""
        if n_node_samples <= 0:
            return 0.0
        return 1.0 / n_node_samples

cdef class MSESplitSensitivity(Sensitivity):
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

cdef class ClassCounterSensitivity(Sensitivity):
    cdef double compute(self, int n_node_samples) noexcept nogil:
        """Sensibilidade da Contagem de Classes: 1"""
        return 1.0

cdef class SumSensitivity(Sensitivity):
    def __cinit__(
        self, 
        double g_max, 
        double g_min
        ):
        cdef double amplitude = fmax(fabs(g_max), fabs(g_min)) 
        self.g_max = g_max
        self.g_min = g_min
        self.amplitude = amplitude
    
    cdef double compute(self, int n_node_samples) noexcept nogil:
        """Sensibilidade de uma soma: max(|G_max|, |G_min|)"""
        return self.amplitude