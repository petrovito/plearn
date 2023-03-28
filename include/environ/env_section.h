#pragma once

#include "rep/call_graph_runner.h"
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

	/**
	 * The class responsible ONLY for executing a call graph.
	 * Has all the resources required provided by some other component.
	 */
	class section_executor {
		public:
			section_executor(const call_graph& cg, const borrowed_ptr<backend_t> backend, 
					hash_map<node_id, tensor_p>& tensors) :
				cg_{cg}, backend_{backend}, tensors_{tensors} {}

			exec_result execute() {
				call_graph_runner runner{cg_};
				runner.run([this] (const op_node& opn) {
					auto& op = opn.op_;
					vector<tensor_p> inputs(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), 
							inputs.begin(), [this](auto id) { return tensors_[id]; });
					tensor_p output = tensors_[opn.out_];
					backend_->exec_op(op, inputs, output);
				});

				vector<tensor_p> outputs(cg_.out_nodes_.size());
				std::transform(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), 
						outputs.begin(), [this](auto id) { return tensors_[id]; });
				return exec_result{outputs};
			}
		private:
			const call_graph& cg_;
			const borrowed_ptr<backend_t> backend_;
			hash_map<node_id, tensor_p>& tensors_;
	};

	/**
	 * A section contains a set of representations and necessary run time variables
	 * required to execute a call graph, and calculate derivatives.
	 *
	 * The section belongs to an execution environment.
	 */
	class env_section {
		public:
			exec_result execute(const exec_params& params) {
				//prepare run
				tensors_ = data_tensors_;
				//add flow tensors and input tensors TODO also output??
				for (auto& [id, node]: cg_.flow_nodes_) {
					tensors_[id] = env_->create_tensor(node.shape_);
				}
				for (unsigned id = 0; id < params.inputs_.size(); ++id) {
					tensors_[id] = params.inputs_[id];
				}
				
				//start run
				section_executor exec{cg_, backend_, tensors_};
				exec_result res = exec.execute();

				return res;
			}
			

		private:
			env_section(borrowed_ptr<exec_env> env, const call_graph& cg) : 
				env_{env}, cg_{cg} {}

			const call_graph& cg_;
			unique_ptr<forward_prop_diff> fp_diff_;
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;

			hash_map<node_id, tensor_p> data_tensors_;
			hash_map<node_id, tensor_p> tensors_;

			friend class exec_env;
			friend class EnvSection_Execute_Test;
	};

}

