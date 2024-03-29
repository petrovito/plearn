#pragma once

#include "backend/cpu/cpu_bw_grad.h"
#include <memory>
#include <cstdlib>

#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_fp_grad.h>

namespace plearn::backend::cpu {

	class cpu_backend : public backend_t {
		public:
			void exec_op(
					const operation& op, 
					const vector<tensor_p>& inputs, 
					tensor_p& output
			) override {
				cpu_tensor* out = static_cast<cpu_tensor*>(output->back());
				vector<cpu_tensor*> in(inputs.size());
				std::transform(inputs.begin(), inputs.end(), in.begin(), 
						[](auto& p) { return static_cast<cpu_tensor*>(p->back()); });

				switch (op.type_) {
					case op_type::noop:
					case op_type::identity: //TODO
						break;
					case op_type::matmul:
						cpu_matmul(op, in, out);
						break;
					case op_type::vecmatmul:
						cpu_vecmatmul(op, in, out);
						break;
					case op_type::matvecmul:
						cpu_matvecmul(op, in, out);
						break;
					case op_type::add:
						cpu_add(op, in, out);
						break;
					case op_type::sub:
						cpu_sub(op, in, out);
						break;
					case op_type::mult:
						cpu_mult(op, in, out);
						break;
					case op_type::square:
						cpu_square(op, in, out);
						break;
					case op_type::reduce_sum:
						cpu_reduce_sum(op, in, out);
						break;
					case op_type::reduce_mean:
						cpu_reduce_mean(op, in, out);
						break;
					case op_type::dot_product:
						cpu_dot_product(op, in, out);
						break;
				}
			}

			unique_ptr<tensor_back_t> create_tensor(const shape_t& shape, 
					tensor_init init=tensor_init::no_init) override {
				auto tens = tens_fac_.allocate(shape);
				switch (init) {
					case tensor_init::no_init:
						break;
					case tensor_init::zero:
						tens->zero();
						break;
					case tensor_init::identity:
						auto buf = tens->get_content()->buf;
						if (shape.rank != 2 || shape.dims[0] != shape.dims[1])
							throw std::runtime_error("Identity tensor must be square"); //TODO extend
						tens->zero();
						for (uint64_t i = 0; i < shape.dims[0]; ++i)
							buf[i * shape.dims[0] + i] = 1.f;
						break;
				}
				return unique_ptr<cpu_tensor>(tens);
			}

			unique_ptr<tensor_back_t> create_tensor(const shape_t& shape, float* data) override {
				return unique_ptr<cpu_tensor>(tens_fac_.create(shape, data));
			}

			unique_ptr<fw_op_diff_backend_t> create_op_fw_diff_backend(
					const operation& op 
			) override {
				switch (op.type_) {
					case op_type::noop:
					case op_type::identity: //TODO
						break;
					case op_type::matmul:
						return std::make_unique<cpu_fp_matmul>();
					case op_type::vecmatmul:
						return std::make_unique<cpu_fp_vecmatmul>();
					case op_type::add:
						break;
					default:
						break;
				}
				throw std::runtime_error("Not implemented");
			}

			unique_ptr<bw_op_diff_backend_t> create_op_bw_diff_backend(
					const operation& op 
			) override {
				switch (op.type_) {
					case op_type::noop:
					case op_type::identity: //TODO
						break;
					case op_type::matmul:
						return std::make_unique<cpu_bw_matmul>();
					case op_type::vecmatmul:
						return std::make_unique<cpu_bw_vecmatmul>();
					case op_type::matvecmul:
						return std::make_unique<cpu_bw_matvecmul>();
					case op_type::square:
						return std::make_unique<cpu_bw_square>();
					case op_type::add:
						return std::make_unique<cpu_bw_add>();
					case op_type::sub:
						return std::make_unique<cpu_bw_sub>();
					case op_type::mult:
						return std::make_unique<cpu_bw_mult>();
					case op_type::reduce_sum:
						return std::make_unique<cpu_bw_reduce_sum>(op.iarg0_);
					case op_type::reduce_mean:
						return std::make_unique<cpu_bw_reduce_mean>(op.iarg0_);
					case op_type::dot_product:
						return std::make_unique<cpu_bw_dot_product>();

				}
				throw std::runtime_error("Not implemented");
			}

		private:
			cpu_tensor_factory tens_fac_;
	};
}

