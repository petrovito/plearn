#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/forward_prop.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>

namespace plearn::env {



	class fp_diff_env {
		public:
			fp_diff_env(
					const call_graph& cg,
					borrowed_ptr<forward_prop_diff> fp_diff,
					borrowed_ptr<backend_t> backend
					) : 
				cg_{cg}, fp_diff_{fp_diff}, backend_{backend} {}

			/**
			 * Initializes the gradients_ map with variable nodes.
			 */
			void init() {
				gradients_.clear();
				//insert variable nodes
				for (auto varn_id : fp_diff_->variable_nodes()) {
					gradients_[varn_id] = {};
					auto& var_shape = cg_.data_nodes_.at(varn_id).shape_;
					gradients_[varn_id].insert({varn_id, {var_shape, var_shape, true, {}}});
				}
				//insert flow and out nodes
				for (auto& [tens_id, _]: cg_.flow_nodes_) {
					if (std::ranges::count(cg_.in_nodes_, tens_id) > 0)
						continue;
					gradients_[tens_id] = {};
					auto& out_shape = cg_.flow_nodes_.at(tens_id).shape_;
					for (auto& var_id: fp_diff_->derivatives().at(tens_id).variable_dependencies()) {
						auto& var_shape = cg_.data_nodes_.at(var_id).shape_;
						auto back_t =backend_->create_tensor(var_shape*out_shape).release();
						gradients_[tens_id][var_id] = gradient{var_shape, out_shape, false, shared_ptr<tensor_back_t>(back_t)};
					}
				}
				
			}

			void calc_diff(const op_node& opn, const vector<tensor_p>& inputs, const tensor_p& output) {
				auto& out_diff = fp_diff_->derivatives().at(opn.out_);
				//TODO check if output is const

				vector<read_ptr<grad_map>> in_diffs(opn.inputs_.size());
				std::transform(opn.inputs_.begin(), opn.inputs_.end(), in_diffs.begin(), 
						[this](node_id id) { return &gradients_.at(id); });

				grad_map& out = gradients_[opn.out_];

				backend_->calc_forward_grad(opn.op_, inputs, output, in_diffs, out);
			}

		borrowed_ptr<grad_system> gradients() { return &gradients_; }

		private:
			const call_graph& cg_;
			borrowed_ptr<forward_prop_diff> fp_diff_;
			borrowed_ptr<backend_t> backend_;

			grad_system gradients_;
	};


	class fp_diff_env_builder {

	};


}

