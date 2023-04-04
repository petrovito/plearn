#pragma once

#include <cstdint>

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
}

