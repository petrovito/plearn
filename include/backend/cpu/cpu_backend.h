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
						[](tensor_p p) { return static_cast<cpu_tensor*>(p->back()); });

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

			void calc_forward_grad(
					const operation& op, 
					const vector<tensor_p>& inputs,
					const tensor_p& output,
					const vector<read_ptr<grad_map>>& in_grads, 
					grad_map& out_grad
			) override {
				//for each variable node calculate the gradient, using the chain rule:
				//   the derivative of the op and the gradients of the inputs
				//NOTE: if a var node is present in the output map, 
				//then some of the inputs are dependant on that var node
				/* for (auto& [varn_id, out_grad] : out_grad) { */
				/* 	for (unsigned i = 0; i < in_grads.size(); ++i) { */
				/* 		if (!in_grads[i]->contains(varn_id)) continue; */
				/* 		auto& in_grad = in_grads[i]->at(varn_id); */
				/* 		//TODO */
				/* 	} */
				/* } */

				//TODO
				switch (op.type_) {
					case op_type::noop:
					case op_type::identity: //TODO
						break;
					case op_type::matmul:
						cpu_matmul_grad(op, inputs, output, in_grads, out_grad);
						break;
					case op_type::vecmatmul:
						break;
					case op_type::add:
						break;
				}
			}

		private:
			cpu_tensor_factory tens_fac_;
	};
}

