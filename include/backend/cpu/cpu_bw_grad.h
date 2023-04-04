#pragma once

#include <cstdint>
#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_op_impl.h>

namespace plearn::backend::cpu {

	class cpu_bw_vecmatmul : public bw_op_diff_backend_t {
		void update_grad(unsigned in_idx,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto other_input_ten = inputs_->at(1-in_idx);
			auto other_input_buf = ((cpu_tensor*)other_input_ten->back())->get_content()->buf;

			auto M = inputs_->at(0)->shape().dims[0];
			auto N = inputs_->at(1)->shape().dims[1];
			const auto out_size = out_outn_grad.out_shape.size();
			if (in_idx == 0) {
				//M -> N -> out_shape
				//in_out_grad(m,n) = other_input_buf(m,n)
#define IN_OUTN_GRAD(m, a) in_outn_grad_buf[m *out_size + a]
#define OUT_OUTN_GRAD(n, a) out_outn_grad_buf[n *out_size + a]
#define OTHER_INPUT_BUF(m,n) other_input_buf[m *N + n]
				for (uint64_t m = 0; m < M; ++m) {
					for (uint64_t n = 0; n < N; ++n) {
						for (uint64_t a = 0; a < out_size; ++a) {
							IN_OUTN_GRAD(m, a) += OUT_OUTN_GRAD(n, a) * OTHER_INPUT_BUF(m,n);
						}
					}
				}
#undef IN_OUTN_GRAD
#undef OUT_OUTN_GRAD
#undef OTHER_INPUT_BUF
			} else { //in_idx == 1
				//MxN -> N -> out_shape
				//in_out_grad(m,n1,n2) = 0 if n1 != n2 else other_input_buf(m)
				//in_outn_grad = in_out_grad * out_outn_grad
				//in_outn_grad(m,n1, a) = sum(n2) in_out_grad(m,n1,n2) * out_outn_grad(n2, a)
				//          = other_input_buf(m) * out_outn_grad(n1, a)
#define IN_OUTN_GRAD(m,n, a) in_outn_grad_buf[m *N*out_size + n *out_size + a]
#define OUT_OUTN_GRAD(n, a) out_outn_grad_buf[n *out_size + a]
#define OTHER_INPUT_BUF(m) other_input_buf[m]
				for (uint64_t m = 0; m < M; ++m) {
					for (uint64_t n = 0; n < N; ++n) {
						for (uint64_t a = 0; a < out_size; ++a) {
							IN_OUTN_GRAD(m,n, a) += OUT_OUTN_GRAD(n, a) * OTHER_INPUT_BUF(m);
						}
					}
				}
#undef IN_OUTN_GRAD
#undef OUT_OUTN_GRAD
#undef OTHER_INPUT_BUF
			}
		}

	};
}

