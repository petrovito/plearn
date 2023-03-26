#pragma once

#include <environ/env_types.h>
#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_ops.h>

namespace plearn::backend::cpu {

	class cpu_backend : public backend_t {
		public:
			void exec_op(
					const operation& op, 
					const vector<tensor_p>& inputs, 
					tensor_p output
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
		private:
	};
}

