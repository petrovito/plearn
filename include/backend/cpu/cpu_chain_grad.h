#pragma once

#include <cstdint>
#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_op_impl.h>

namespace plearn::backend::cpu {

	inline void cpu_matmul_grad(
			const operation&, 
			const vector<tensor_p>& inputs,
			const tensor_p&,
			const vector<read_ptr<grad_map>>& in_grads, 
			grad_map& out_grads
		) {
		
			auto& first_shape = inputs[0]->shape();
			auto& second_shape = inputs[1]->shape();
			auto M = first_shape.dims[0];
			auto N = first_shape.dims[1];
			auto K = second_shape.dims[1];

		{
			//FIRST MATRIX
			unsigned i = 0;
			unsigned other = 1 - i;
			auto second_buf = ((cpu_tensor*)inputs[other]->back())->get_content()->buf;
			//shape of first=i: [m,n] shape of second: [n,k]
			//i_out_grad[m1,n, m2,k]=0 if m1!=m2, second[n,k] otherwise
			for (auto& [varn_id, var_out_grad] : out_grads) {
				if (!in_grads[i]->contains(varn_id)) continue;
				//shape of var: [A]
				auto& var_i_grad = in_grads[i]->at(varn_id);
				auto var_i_grad_buf = ((cpu_tensor*)var_i_grad.back_.get())->get_content()->buf;
				auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
				//shape of var_i_grad: [A, M,N]
				//shape of i_out_grad: [M,N, M,K]
				//shape of var_out_grad: [A, M,K]
				//var_out_grad[a, m2,k] = sum(m1,n) var_i_grad[a, m1,n] * i_out_grad[m1,n, m2,k]
				//           = sum(n) var_i_grad[a, m2,n] * other[n,k]
#define VAR_OUT_GRAD(a,m2,k) var_out_grad_buf[a *M*K + m2 *K + k]
#define VAR_I_GRAD(a,m2,n) var_i_grad_buf[a *M*N + m2 *N + n]
				for (uint64_t a = 0; a < var_i_grad.in_shape.size(); a++) {
					for (uint64_t m2 = 0; m2 < M; m2++) {
						for (uint64_t n = 0; n < N; n++) {
							for (uint64_t k = 0; k < K; k++) {
								VAR_OUT_GRAD(a,m2,k) += VAR_I_GRAD(a,m2,n) * second_buf[n *K + k];
							}
						}
					}
				}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
			}
		}
		{
			//SECOND MATRIX
			unsigned i = 1;
			unsigned other = 1 - i;
			auto first_buf = ((cpu_tensor*)inputs[other]->back())->get_content()->buf;
			//shape of first: [m,n] shape of second=i: [n,k]
			//i_out_grad[n,k1, m,k2]=0 if k1!=k2, first[m,n] otherwise
			for (auto& [varn_id, var_out_grad] : out_grads) {
				if (!in_grads[i]->contains(varn_id)) continue;
				//shape of var: [A]
				auto& var_i_grad = in_grads[i]->at(varn_id);
				auto var_i_grad_buf = ((cpu_tensor*)var_i_grad.back_.get())->get_content()->buf;
				auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
				//shape of var_i_grad: [A, N,K]
				//shape of i_out_grad: [N,K, M,K]
				//shape of var_out_grad: [A, M,K]
				//var_out_grad[a, m,k2] = sum(k1,n) var_i_grad[a, n,k1] * i_out_grad[n,k1, m,k2]
				//		   = sum(n) var_i_grad[a, n,k2] * first[m,n]
#define VAR_OUT_GRAD(a,m,k2) var_out_grad_buf[a *M*K + m *K + k2]
#define VAR_I_GRAD(a,n,k2) var_i_grad_buf[a *N*K + n *K + k2]
				for (uint64_t a = 0; a < var_i_grad.in_shape.size(); a++) {
					for (uint64_t m = 0; m < M; m++) {
						for (uint64_t n = 0; n < N; n++) {
							for (uint64_t k2 = 0; k2 < K; k2++) {
								VAR_OUT_GRAD(a,m,k2) += VAR_I_GRAD(a,n,k2) * first_buf[m *N + n];
							}
						}
					}
				}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
			}
		}
	}


	inline void cpu_vecmatmul_grad(
			const operation&, 
			const vector<tensor_p>& inputs,
			const tensor_p&,
			const vector<read_ptr<grad_map>>& in_grads, 
			grad_map& out_grads
		) {
		
			auto& first_shape = inputs[0]->shape();
			auto& second_shape = inputs[1]->shape();
			auto M = first_shape.dims[0];
			auto N = second_shape.dims[1];

			{
				//FIRST VECTOR
				unsigned i = 0;
				unsigned other = 1 - i;
				auto second_buf = ((cpu_tensor*)inputs[other]->back())->get_content()->buf;
				//shape of first=i: [m] shape of second: [m,n]
				//i_out_grad[m, n]=second[m,n]
				for (auto& [varn_id, var_out_grad] : out_grads) {
					if (!in_grads[i]->contains(varn_id)) continue;
					//shape of var: [A]
					auto& var_i_grad = in_grads[i]->at(varn_id);
					auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
					if (var_i_grad.identity) {
						//shape of var_i_grad: [M, M]
						//shape of i_out_grad: [M, N]
						//var_out_grad[m, n] = second[m,n]
						for (uint64_t m = 0; m < M; m++) {
							for (uint64_t n = 0; n < N; n++) {
								var_out_grad_buf[m *N + n] += second_buf[m *N + n];
							}
						}
						continue;
					}
					auto var_i_grad_buf = ((cpu_tensor*)var_i_grad.back_.get())->get_content()->buf;
					//shape of var_i_grad: [A, M]
					//shape of i_out_grad: [M, N]
					//shape of var_out_grad: [A, N]
					//var_out_grad[a, n] = sum(m) var_i_grad[a, m] * i_out_grad[m, n]
					//		   = sum(m) var_i_grad[a, m] * second[m,n]
#define VAR_OUT_GRAD(a,n) var_out_grad_buf[a *N + n]
#define VAR_I_GRAD(a,m) var_i_grad_buf[a *M + m]
					for (uint64_t a = 0; a < var_i_grad.in_shape.size(); a++) {
						for (uint64_t m = 0; m < M; m++) {
							for (uint64_t n = 0; n < N; n++) {
								VAR_OUT_GRAD(a,n) += VAR_I_GRAD(a,m) * second_buf[m *N + n];
							}
						}
					}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
				}
			}		
			{
				//SECOND MATRIX
				unsigned i = 1;
				unsigned other = 1 - i;
				auto first_buf = ((cpu_tensor*)inputs[other]->back())->get_content()->buf;
				//shape of first: [m] shape of second=i: [m,n]
				//i_out_grad[m,n1, n2]=0 if n1!=n2, first[m] otherwise
				for (auto& [varn_id, var_out_grad] : out_grads) {
					if (!in_grads[i]->contains(varn_id)) continue;
					//shape of var: [A]
					auto& var_i_grad = in_grads[i]->at(varn_id);

					//shape of var_i_grad: [A, M,N]
					//shape of i_out_grad: [M,N, N]
					//shape of var_out_grad: [A, N]
					auto var_out_grad_buf = ((cpu_tensor*)var_out_grad.back_.get())->get_content()->buf;
					if (var_i_grad.identity) {
						//var_i_grad[m0,n0, m1,n1] = 1 if n0==n1 and m0==m1, 0 otherwise
						//var_out_grad[m0,n0, n2] = sum(m1, n1) var_i_grad[m0,n0, m1,n1] * i_out_grad[m1,n1, n2]
						//			= i_out_grad[m0,n0, n2] = 0 if n0!=n2, first[m0] otherwise
						for (uint64_t m0 = 0; m0 < M; m0++) {
							for (uint64_t n0 = 0; n0 < N; n0++) {
								auto n2 = n0;
								var_out_grad_buf[m0 *N*N + n0 *N + n2] += first_buf[m0];
							}
						}
						continue;
					}

					auto var_i_grad_buf = ((cpu_tensor*)var_i_grad.back_.get())->get_content()->buf;
					//var_out_grad[a, n2] = sum(m,n1) var_i_grad[a, m,n1] * i_out_grad[m,n1, n2]
					//		   = sum(m) var_i_grad[a, m,n2] * first[m]
#define VAR_OUT_GRAD(a,n2) var_out_grad_buf[a *N + n2]
#define VAR_I_GRAD(a,m,n2) var_i_grad_buf[a *M*N + m *N + n2]
					for (uint64_t a = 0; a < var_i_grad.in_shape.size(); a++) {
						for (uint64_t m = 0; m < M; m++) {
							for (uint64_t n2 = 0; n2 < N; n2++) {
								VAR_OUT_GRAD(a,n2) += VAR_I_GRAD(a,m,n2) * first_buf[m];
							}
						}
					}
#undef VAR_OUT_GRAD
#undef VAR_I_GRAD
				}
			}
	}
}

