#pragma once

#include <memory>
#include <cstdlib>

#include <rep/rep_types.h>
#include <environ/env_types.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_ops.h>
#include <backend/cpu/cpu_chain_grad.h>

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
					case op_type::add:
						cpu_add(op, in, out);
						break;
				}
			}

			unique_ptr<tensor_back_t> create_tensor(const shape_t& s) override {
				auto tens = tens_fac_.allocate(s);
				return unique_ptr<cpu_tensor>(tens);
			}

			unique_ptr<fp_op_diff_backend_t> create_op_diff_backend(
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
				}
			}

		private:
			cpu_tensor_factory tens_fac_;
	};
}

