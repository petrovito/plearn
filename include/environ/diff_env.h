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



	class fp_diff_env {
		public:
			void calc_diff(const op_node& opn, const vector<tensor_p>& inputs, const tensor_p& output) {
				auto& out_diff = fp_diff_->derivatives().at(opn.out_);
				//TODO check if output is const

				vector<read_ptr<grad_map>> in_diffs(opn.inputs_.size());
				std::transform(opn.inputs_.begin(), opn.inputs_.end(), in_diffs.begin(), 
						[this](node_id id) { return &gradients_.at(id); });

				if (!gradients_.contains(opn.out_))
					gradients_[opn.out_] = {};
				grad_map& out = gradients_[opn.out_];

				backend_->calc_forward_grad(opn.op_, inputs, output, in_diffs, out);
			}

		private:
			borrowed_ptr<forward_prop_diff> fp_diff_;
			borrowed_ptr<backend_t> backend_;

			grad_system gradients_;
	};


}

