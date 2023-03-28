#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/forward_prop.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>

namespace plearn::env {


	struct grad_wrt {
		node_id input;
		node_id output;

		bool operator==(const grad_wrt& other) const = default;
	};

	/**
	 * Hashing function for grad_wrt
	 */
	struct grad_wrt_hash {
		std::size_t operator()(const grad_wrt& gw) const {
			return std::hash<node_id>{}(gw.input) ^ std::hash<node_id>{}(gw.output);
		}
	};

	class fp_diff_env {
		public:
			void calc_diff(const op_node& opn, const vector<tensor_p>& inputs, const tensor_p& output) {
				auto& out_diff = fp_diff_->derivatives().at(opn.out_);
				//calculate direct gradients
				for (unsigned i = 0; i < opn.inputs_.size(); ++i) {
					auto inn_id = opn.inputs_[i];
					auto& grad_tens = gradients_.at({inn_id, opn.out_});
					backend_->calc_grad(opn.op_, i, grad_tens, inputs, output);
					//TODO this is only necessary if input is dependant on any variable
					//TODO 2: take care of trivial dependencies (identity and independent)
				}

				//calculate indirect gradients using chain rule
				for (auto& varn_id: fp_diff_->variable_nodes()) {
					//if output doesnt depend on var, nothing to do
					if (!out_diff.depends_on(varn_id)) continue;

					auto& varn_out_grad = gradients_.at({varn_id, opn.out_});

					operation multiply_and_add{op_type::matmul, output_modify_t::add};
					for (auto inn_id: opn.inputs_) {
						auto& inn_diff = fp_diff_->derivatives().at(inn_id);
						//if input doesnt depend on var, nothing to do
						if (!inn_diff.depends_on(varn_id)) continue;

						auto& inn_out_grad = gradients_.at({inn_id, opn.out_});
						auto& varn_inn_grad = gradients_.at({varn_id, inn_id});

						//TODO: take care of trivial dependencies (identity and independent)
						backend_->exec_op(multiply_and_add, {varn_inn_grad, inn_out_grad}, varn_out_grad);
					}
										
				}
								
			}
		private:
			unordered_map<grad_wrt, tensor_p, grad_wrt_hash> gradients_;
			borrowed_ptr<forward_prop_diff> fp_diff_;
			borrowed_ptr<backend_t> backend_;
	};


}

