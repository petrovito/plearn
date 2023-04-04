#pragma once

#include <cstdint>
#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_op_impl.h>

namespace plearn::backend::cpu {

class cpu_fp_matmul : public fw_op_diff_backend_t {
	void update_grad(unsigned input_idx,
			const gradient& var_in_grad,  gradient& var_out_grad) override {
		auto var_in_grad_buf = ((cpu_tensor*)var_in_grad.back_.get())->get_content()->buf;
		auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
		auto other_input_buf = ((cpu_tensor*)inputs_->at(1-input_idx)->back())->get_content()->buf;
		auto& in1_shape = inputs_->at(0)->shape();
		auto& in2_shape = inputs_->at(1)->shape();
		auto M = in1_shape.dims[0];
		auto N = in1_shape.dims[1];
		auto K = in2_shape.dims[1];
		if (input_idx == 0) {
			//shape of var_i_grad: [A, M,N]
			//shape of i_out_grad: [M,N, M,K]
			//shape of var_out_grad: [A, M,K]
			//var_out_grad[a, m2,k] = sum(m1,n) var_i_grad[a, m1,n] * i_out_grad[m1,n, m2,k]
			//           = sum(n) var_i_grad[a, m2,n] * other[n,k]
#define VAR_OUT_GRAD(a,m2,k) var_out_grad_buf[a *M*K + m2 *K + k]
#define VAR_I_GRAD(a,m2,n) var_in_grad_buf[a *M*N + m2 *N + n]
			for (uint64_t a = 0; a < var_in_grad.in_shape.size(); a++) {
				for (uint64_t m2 = 0; m2 < M; m2++) {
					for (uint64_t n = 0; n < N; n++) {
						for (uint64_t k = 0; k < K; k++) {
							VAR_OUT_GRAD(a,m2,k) += VAR_I_GRAD(a,m2,n) * other_input_buf[n *K + k];
						}
					}
				}
			}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
		} else { //input_idx == 1
#define VAR_OUT_GRAD(a,m,k2) var_out_grad_buf[a *M*K + m *K + k2]
#define VAR_I_GRAD(a,n,k2) var_in_grad_buf[a *N*K + n *K + k2]
			for (uint64_t a = 0; a < var_in_grad.in_shape.size(); a++) {
				for (uint64_t m = 0; m < M; m++) {
					for (uint64_t n = 0; n < N; n++) {
						for (uint64_t k2 = 0; k2 < K; k2++) {
							VAR_OUT_GRAD(a,m,k2) += VAR_I_GRAD(a,n,k2) * other_input_buf[m *N + n];
						}
					}
				}
			}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
		}
	}

	void update_grad_with_identity(unsigned input_idx, 
			gradient& var_out_grad) override { 
		auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
		auto other_input_buf = ((cpu_tensor*)inputs_->at(1-input_idx)->back())->get_content()->buf;
		auto& in1_shape = inputs_->at(0)->shape();
		auto& in2_shape = inputs_->at(1)->shape();
		auto M = in1_shape.dims[0];
		auto N = in1_shape.dims[1];
		auto K = in2_shape.dims[1];

		if (input_idx == 0) {

#define VAR_OUT_GRAD(m1,n,m2,k) var_out_grad_buf[m1 * N * M * K + n * M * K + m2 * K + k]
			for (uint64_t m1 = 0; m1 < M; m1++) {
				for (uint64_t n = 0; n < N; n++) {
					for (uint64_t k = 0; k < K; k++) {
						auto m2 = m1;
						VAR_OUT_GRAD(m1,n,m2,k) = other_input_buf[n*K + k];

					}
				}
			}
#undef VAR_OUT_GRAD
		} else { //input_idx == 1
#define VAR_OUT_GRAD(n,k1,m,k2) var_out_grad_buf[n *M*K*K + k1 *M*K + m*K + k2]
			for (uint64_t n = 0; n < N; n++) {
				for (uint64_t m = 0; m < M; m++) {
					for (uint64_t k1 = 0; k1 < K; k1++) {
						auto k2 = k1;
						VAR_OUT_GRAD(n,k1,m,k2) += other_input_buf[m *N + n];
					}
				}
			}
#undef VAR_OUT_GRAD

		}

	}
};



class cpu_fp_vecmatmul : public fw_op_diff_backend_t {
	void update_grad(unsigned input_idx,
			const gradient& var_in_grad,  gradient& var_out_grad) override {
		auto var_in_grad_buf = ((cpu_tensor*)var_in_grad.back_.get())->get_content()->buf;
		auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
		auto other_input_ten = inputs_->at(1-input_idx);
		auto other_input_buf = ((cpu_tensor*)other_input_ten->back())->get_content()->buf;
		auto M = var_in_grad.out_shape.dims[0];
		auto N = var_out_grad.out_shape.dims[0];
		if (input_idx == 0) {
			//shape of var_i_grad: [A, M]
			//shape of i_out_grad: [M, N]
			//shape of var_out_grad: [A, N]
			//var_out_grad[a, n] = sum(m) var_i_grad[a, m] * i_out_grad[m, n]
			//		   = sum(m) var_i_grad[a, m] * second[m,n]
#define VAR_OUT_GRAD(a,n) var_out_grad_buf[a *N + n]
#define VAR_I_GRAD(a,m) var_in_grad_buf[a *M + m]
			for (uint64_t a = 0; a < var_in_grad.in_shape.size(); a++) {
				for (uint64_t m = 0; m < M; m++) {
					for (uint64_t n = 0; n < N; n++) {
						VAR_OUT_GRAD(a,n) += VAR_I_GRAD(a,m) * other_input_buf[m *N + n];
					}
				}
			}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
		} else { //input_idx == 1
				 //var_out_grad[a, n2] = sum(m,n1) var_i_grad[a, m,n1] * i_out_grad[m,n1, n2]
				 //		   = sum(m) var_i_grad[a, m,n2] * first[m]
#define VAR_OUT_GRAD(a,n2) var_out_grad_buf[a *N + n2]
#define VAR_I_GRAD(a,m,n2) var_in_grad_buf[a *M*N + m *N + n2]
			for (uint64_t a = 0; a < var_in_grad.in_shape.size(); a++) {
				for (uint64_t m = 0; m < M; m++) {
					for (uint64_t n2 = 0; n2 < N; n2++) {
						VAR_OUT_GRAD(a,n2) += VAR_I_GRAD(a,m,n2) * other_input_buf[m];
					}
				}
			}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
		}
	}


	void update_grad_with_identity(unsigned input_idx, 
			gradient& var_out_grad) override {
		auto other_input_ten = inputs_->at(1-input_idx);
		auto other_input_buf = ((cpu_tensor*)other_input_ten->back())->get_content()->buf;
		auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
		auto M = inputs_->at(0)->shape().dims[0];
		auto N = var_out_grad.out_shape.dims[0];
		if (input_idx == 0) {
			for (uint64_t m = 0; m < M; m++) {
				for (uint64_t n = 0; n < N; n++) {
					var_out_grad_buf[m *N + n] += other_input_buf[m *N + n];
				}
			}
		} else { //input_idx == 1
			for (uint64_t m0 = 0; m0 < M; m0++) {
				for (uint64_t n0 = 0; n0 < N; n0++) {
					auto n2 = n0;
					var_out_grad_buf[m0 *N*N + n0 *N + n2] += other_input_buf[m0];
				}
			}
		}
	}
};

}

