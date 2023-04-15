#pragma once

#include <cstdint>
#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_op_impl.h>

namespace plearn::backend::cpu {

	class cpu_bw_vecmatmul : public bw_op_diff_backend_t {
		public: 
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
				for (uint64_t a = 0; a < out_size; ++a) {
					_cpu_matvecmul(other_input_buf, out_outn_grad_buf + a, in_outn_grad_buf + a,
							M, N);
				}
			} else { //in_idx == 1
				for (uint64_t a = 0; a < out_size; ++a) {
					_cpu_matmul(other_input_buf, out_outn_grad_buf + a, in_outn_grad_buf + a,
							M, 1, N, true);
				}
			}
		}
	};

	class cpu_bw_matvecmul : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned in_idx,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto other_input_ten = inputs_->at(1-in_idx);
			auto other_input_buf = ((cpu_tensor*)other_input_ten->back())->get_content()->buf;

			auto M = inputs_->at(0)->shape().dims[0];
			auto N = inputs_->at(0)->shape().dims[1];
			const auto out_size = out_outn_grad.out_shape.size();
			if (in_idx == 0) {
				for (uint64_t a = 0; a < out_size; ++a) {
					_cpu_matmul(out_outn_grad_buf + a, other_input_buf, in_outn_grad_buf + a,
							M, 1, N, true);
				}
			} else { //in_idx == 1
				for (uint64_t a = 0; a < out_size; ++a) {
					_cpu_vecmatmul(out_outn_grad_buf + a, other_input_buf, in_outn_grad_buf + a,
							M, N);
				}
			}
		}
	};

	class cpu_bw_square : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned, 
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto in_buf = ((cpu_tensor*)inputs_->at(0)->back())->get_content()->buf;
			auto& shape = in_outn_grad.in_shape;
			auto size = shape.size();
			auto& outn_shape = out_outn_grad.out_shape;
			auto outn_size = outn_shape.size();
			for (uint64_t i = 0; i < size; ++i) {
				for (uint64_t j = 0; j < outn_size; ++j) {
					in_outn_grad_buf[i *outn_size +j] += 2 * in_buf[i] * out_outn_grad_buf[i *outn_size + j];
				}
			}
		}
	};

	class cpu_bw_add : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto& shape = in_outn_grad.in_shape;
			auto size = shape.size();
			auto& outn_shape = out_outn_grad.out_shape;
			auto outn_size = outn_shape.size();
			for (uint64_t i = 0; i < size; ++i) {
				for (uint64_t j = 0; j < outn_size; ++j) {
					in_outn_grad_buf[i *outn_size +j] += out_outn_grad_buf[i *outn_size + j];
				}
			}
		}
	};


	class cpu_bw_sub : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned in_idx,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto& shape = in_outn_grad.in_shape;
			auto size = shape.size();
			auto& outn_shape = out_outn_grad.out_shape;
			auto outn_size = outn_shape.size();
			if (in_idx == 0) {
				for (uint64_t i = 0; i < size; ++i) {
					for (uint64_t j = 0; j < outn_size; ++j) {
						in_outn_grad_buf[i *outn_size +j] += out_outn_grad_buf[i *outn_size + j];
					}
				}
			} else {
				for (uint64_t i = 0; i < size; ++i) {
					for (uint64_t j = 0; j < outn_size; ++j) {
						in_outn_grad_buf[i *outn_size +j] -= out_outn_grad_buf[i *outn_size + j];
					}
				}
			}
		}
	};

	class cpu_bw_mult : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned in_idx,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto& shape = in_outn_grad.in_shape;
			auto size = shape.size();
			auto& outn_shape = out_outn_grad.out_shape;
			auto outn_size = outn_shape.size();
			auto other_input_buf = ((cpu_tensor*)inputs_->at(1 - in_idx)->back())->get_content()->buf;
			for (uint64_t i = 0; i < size; ++i) {
				for (uint64_t j = 0; j < outn_size; ++j) {
					in_outn_grad_buf[i *outn_size +j] += other_input_buf[i] * out_outn_grad_buf[i *outn_size + j];
				}
			}
		}
	};

	class cpu_bw_dot_product : public bw_op_diff_backend_t {
		public:
		void update_grad(unsigned in_idx,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto& shape = in_outn_grad.in_shape;
			auto size = shape.size();
			auto& outn_shape = out_outn_grad.out_shape;
			auto outn_size = outn_shape.size();
			auto other_input_buf = ((cpu_tensor*)inputs_->at(1 - in_idx)->back())->get_content()->buf;
			for (uint64_t i = 0; i < size; ++i) {
				for (uint64_t j = 0; j < outn_size; ++j) {
					in_outn_grad_buf[i *outn_size +j] += other_input_buf[i] * out_outn_grad_buf[j];
				}
			}
		}
	};

	class cpu_bw_reduce_sum : public bw_op_diff_backend_t {
		public:
		cpu_bw_reduce_sum(unsigned axis) : axis(axis) {}
		void update_grad(unsigned,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto size = in_outn_grad.in_shape.size();
			uint64_t M = 1;
			for (unsigned i = 0; i < axis; ++i) {
				M *= in_outn_grad.in_shape.dims[i];
			}
			uint64_t AXIS_SIZE = in_outn_grad.in_shape.dims[axis];
			uint64_t N = 1;
			for (unsigned i = axis + 1; i < in_outn_grad.in_shape.dims.size(); ++i) {
				N *= in_outn_grad.in_shape.dims[i];
			}
			for (uint64_t m = 0; m < M; ++m) {
				for (uint64_t a = 0; a < AXIS_SIZE; ++a) {
					for (uint64_t n = 0; n < N; ++n) {
						in_outn_grad_buf[m * AXIS_SIZE*N + a *N + n] += out_outn_grad_buf[m *N + n];
					}
				}
			}
		}
			
		private:
		unsigned axis;
	};

	class cpu_bw_reduce_mean : public bw_op_diff_backend_t {
		public:
		cpu_bw_reduce_mean(unsigned axis) : axis(axis) {}
		void update_grad(unsigned,
				const gradient& out_outn_grad, gradient& in_outn_grad) override {
			auto out_outn_grad_buf = ((cpu_tensor*)out_outn_grad.back_.get())->get_content()->buf;
			auto in_outn_grad_buf = ((cpu_tensor*)in_outn_grad.back_.get())->get_content()->buf;
			auto size = in_outn_grad.in_shape.size();
			uint64_t M = 1;
			for (unsigned i = 0; i < axis; ++i) {
				M *= in_outn_grad.in_shape.dims[i];
			}
			uint64_t AXIS_SIZE = in_outn_grad.in_shape.dims[axis];
			uint64_t N = 1;
			for (unsigned i = axis + 1; i < in_outn_grad.in_shape.dims.size(); ++i) {
				N *= in_outn_grad.in_shape.dims[i];
			}
			for (uint64_t m = 0; m < M; ++m) {
				for (uint64_t a = 0; a < AXIS_SIZE; ++a) {
					for (uint64_t n = 0; n < N; ++n) {
						in_outn_grad_buf[m * AXIS_SIZE*N + a *N + n] += out_outn_grad_buf[m *N + n] / AXIS_SIZE;
					}
				}
			}
		}
			
		private:
		unsigned axis;
	};

}

