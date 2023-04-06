#pragma once

#include <cstdint>
#include <vector>

namespace plearn::backend::cpu {
	
	inline void _cpu_matmul(float* A, float* B,
			float* C, uint64_t m, uint64_t n, uint64_t k) {
		
		for (uint64_t i = 0; i < m; ++i) {
			for (uint64_t l = 0; l < n; ++l) {
				for (uint64_t j = 0; j < k; ++j) {
					C[i*k +j] += A[i*n +l] * B[l*k +j];
				}
			}
		}
	}


	inline void _cpu_vecmatmul(float* A, float*B, float* C, uint64_t m, uint64_t n) {
		for (uint64_t i = 0; i < m; ++i) {
			for (uint64_t j = 0; j < n; ++j) {
				C[j] += A[i] * B[i*n + j];
			}
		}
	}

	inline void _cpu_matvecmul(float* A, float*B, float* C, uint64_t m, uint64_t n) {
		for (uint64_t i = 0; i < m; ++i) {
			for (uint64_t l = 0; l < n; ++l) {
				C[i] += A[i*n +l] * B[l];
			}
		}
	}

	inline void _cpu_add(float* A, float* B, float* C, uint64_t len) {
		for (uint64_t i = 0; i < len; i++) {
			C[i] = A[i] + B[i];
		}
	}

	inline void _cpu_sub(float* A, float* B, float* C, uint64_t len) {
		for (uint64_t i = 0; i < len; i++) {
			C[i] = A[i] - B[i];
		}
	}

	inline void _cpu_mult(float* A, float* B, float* C, uint64_t len) {
		for (uint64_t i = 0; i < len; i++) {
			C[i] = A[i] * B[i];
		}
	}

	inline void _cpu_square(float* A, float* B, uint64_t len) {
		for (uint64_t i = 0; i < len; i++) {
			B[i] = A[i] * A[i];
		}
	}

	inline void _cpu_dot_product(float* A, float* B, float* C, uint64_t len) {
		for (uint64_t i = 0; i < len; i++) {
			C[0] += A[i] * B[i];
		}
	}

	inline void _cpu_reduce_sum(float* A, float* C, unsigned axis, std::vector<uint64_t> shape) {
		uint64_t m = 1;
		uint64_t n = 1;
		for (unsigned i = 0; i < axis; i++) {
			m *= shape[i];
		}
		for (unsigned i = axis + 1; i < shape.size(); i++) {
			n *= shape[i];
		}
		auto axis_len = shape[axis];
		for (uint64_t i = 0; i < m; i++) {
			for (uint64_t k = 0; k < axis_len; k++) {
				for (uint64_t j = 0; j < n; j++) {
					C[i*n + j] += A[i *axis_len*n + k*n + j];
				}
			}
		}
	}

	inline void _cpu_reduce_mean(float* A, float* C, unsigned axis, std::vector<uint64_t> shape) {
		uint64_t m = 1;
		uint64_t n = 1;
		for (unsigned i = 0; i < axis; i++) {
			m *= shape[i];
		}
		for (unsigned i = axis + 1; i < shape.size(); i++) {
			n *= shape[i];
		}
		auto axis_len = shape[axis];
		for (uint64_t i = 0; i < m; i++) {
			for (uint64_t k = 0; k < axis_len; k++) {
				for (uint64_t j = 0; j < n; j++) {
					C[i*n + j] += A[i *axis_len*n + k*n + j] / axis_len;
				}
			}
		}
	}
}

