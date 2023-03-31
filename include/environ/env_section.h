#pragma once

#include <bits/ranges_algo.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/call_graph_runner.h>
#include <algorithm>
#include <rep/forward_prop.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>
#include <environ/diff_env.h>

namespace plearn::env {


	/**
	 * Holds resourcers for ONE execution of a call graph.
	 * Resources are managed by another component.
	 *
	 * Resources means internal tensors, that are not provided by an external component.
	 */
	struct section_resources {
		hash_map<node_id, tensor_p> internal_tensors_;
	};

	/**
	 * Container for all the tensors that are required for call graph executions.
	 */
	struct section_exec_tensors {
		template<typename... Args>
		section_exec_tensors(
				hash_map<node_id, tensor_p>& tensors,
				Args&&... other_tens

				) :
			tensors_{tensors} {
				(tensors_.insert(other_tens.begin(), other_tens.end()), ...);

			}
		hash_map<node_id, tensor_p> tensors_;

		tensor_p& operator[](node_id id) { return tensors_[id]; }
	};

	/**
	 * The class responsible ONLY for executing a call graph.
	 * Has all the resources required provided by some other component.
	 */
	class section_executor {
		public:
			section_executor(const exec_params& params,
					const call_graph& cg, borrowed_ptr<fp_diff_env> diff_env,
					const borrowed_ptr<backend_t> backend, 
					section_exec_tensors& tensors) : 
				params_{params}, cg_{cg}, diff_env_{diff_env}, 
				backend_{backend}, tensors_{tensors} {}

			void execute() {
				call_graph_runner runner{cg_};
				runner.run([this] (const op_node& opn) {
					auto& op = opn.op_;
					//collect inputs and outputs
					vector<tensor_p> inputs(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), 
							inputs.begin(), [this](auto id) { return tensors_[id]; });
					tensor_p output = tensors_[opn.out_];
					//execute operation
					backend_->exec_op(op, inputs, output);
					//calculate derivatives if required
					if (params_.calc_diffs) {
						diff_env_->calc_diff(opn, inputs, output);
					}
				});

				vector<tensor_p> outputs(cg_.out_nodes_.size());
				std::transform(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), 
						outputs.begin(), [this](auto id) { return tensors_[id]; });
			}
		private:
			const exec_params& params_;
			const call_graph& cg_;
			borrowed_ptr<fp_diff_env> diff_env_;
			borrowed_ptr<backend_t> backend_;
			section_exec_tensors& tensors_;
	};

	/**
	 * A section is a component that is responsible for managing resources for operations 
	 * on a call graph, i.e. execution/differentials.
	 * Resource typically means memory.
	 */
	class env_section {
		public:
			env_section(borrowed_ptr<exec_env> env, borrowed_ptr<backend_t> backend,
					const call_graph& cg,
					const hash_map<node_id, tensor_p>& data_tensors) :  
				cg_{cg}, env_{env}, backend_{backend}, data_tensors_{data_tensors} {
					allocate_internal_tensors();
					create_fp_diff();
				}

			exec_result execute(exec_params& params) {
				//prepare run
				reset_internal_tensors();
				if (params.calc_diffs) {
					fp_diff_env_->reset();
				}
				section_exec_tensors tensors{resources_.internal_tensors_, data_tensors_,
					params.inputs_, params.outputs_};
				
				//start run
				section_executor exec{params, cg_, fp_diff_env_.get(), backend_, tensors};
				exec.execute();

				return {.success=true, .grads=fp_diff_env_->gradients()};
			}
			

		private:
			void allocate_internal_tensors() {
				for (auto intn_id: cg_.internal_nodes_) {
					auto& node = cg_.flow_nodes_.at(intn_id);
					resources_.internal_tensors_[intn_id] = env_->create_tensor(node.shape_);
				}
			}

			void reset_internal_tensors() {
				for (auto& [id, tens]: resources_.internal_tensors_) {
					tens->back()->zero();
				}
			}

			void create_fp_diff() {
				forward_prop_diff_builder builder{cg_};
				fp_diff_ = builder
					.all_data_nodes()
					.find_depending_tensors()
					.build();
				fp_diff_env_ = std::make_unique<fp_diff_env>(cg_, fp_diff_.get(), backend_);
				fp_diff_env_->init();
			}

			const call_graph& cg_;
			unique_ptr<forward_prop_diff> fp_diff_;
			unique_ptr<fp_diff_env> fp_diff_env_;
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;

			hash_map<node_id, tensor_p> data_tensors_;
			section_resources resources_;

			friend class EnvSection_Execute_Test;
	};

}

