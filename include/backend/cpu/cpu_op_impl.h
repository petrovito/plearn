#pragma once

#include <immintrin.h>

namespace plearn {
	
	inline void _cpu_matmul(float* A, float* B,
			float* C, int m, int n, int k) {
		
		for (int i = 0; i < m; ++i) {
			for (int l = 0; l < n; ++l) {
				for (int j = 0; j < k; ++j) {
					C[i*k +j] += A[i*n +l] * B[l*k +j];
				}
			}
		}
	}

	//unused
	inline void _cpu_matmul_avx(float* A, float*B, float* C, int m, int n, int k) {
		//TODO only handles multiples of 8
		for (int i = 0; i < m; i++) {
			for (int l = 0; l < k; l += 16) {
				auto sumA = _mm256_setzero_ps();
				auto sumB = _mm256_setzero_ps();
				for (int j = 0; j < n; j++) {
					auto bc_mat1 = _mm256_set1_ps(A[i*n +j]);
					auto vecA_mat2 = _mm256_load_ps(&B[j*k +l]);
					auto vecB_mat2 = _mm256_load_ps(&B[j*k +l +8]);
					auto prodA = _mm256_mul_ps(bc_mat1, vecA_mat2);
					auto prodB = _mm256_mul_ps(bc_mat1, vecB_mat2);
					sumA = _mm256_add_ps(sumA, prodA);
					sumB = _mm256_add_ps(sumB, prodB);
				}
				_mm256_store_ps(&C[i*k +l], sumA);
				_mm256_store_ps(&C[i*k +l +8], sumB);
			}
		}
	}



	inline void _cpu_matvecmul(float* A, float*B, float* C, int m, int n) {
		for (int i = 0; i < m; ++i) {
			for (int l = 0; l < n; ++l) {
					C[i] += A[i*n +l] * B[l];
			}
		}
	}

	inline void _cpu_add(float* A, float* B, float* C, int len) {
		for (int i = 0; i < len; i++) {
			C[i] = A[i] + B[i];
		}
	}
}

