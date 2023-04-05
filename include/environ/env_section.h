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
#include <rep/diff_info.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>
#include <environ/fp_diff_env.h>
#include "environ/bw_diff_env.h"

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
	 * The class responsible ONLY for executing a call graph.
	 * Has all the resources required provided by some other component.
	 */
	class section_executor {
		public:
			section_executor(const exec_params& params,
					const call_graph& cg, 
					const borrowed_ptr<backend_t> backend, 
					section_exec_tensors& tensors) : 
				params_{params}, cg_{cg},
				backend_{backend}, tensors_{tensors} {}

			void execute() {
				call_graph_forward_runner runner{cg_};
				runner.run([this] (const op_node& opn) {
					auto& op = opn.op_;
					//collect inputs and outputs
					vector<tensor_p> inputs(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), 
							inputs.begin(), [this](auto id) { return tensors_[id]; });
					tensor_p output = tensors_[opn.out_];
					//execute operation
					backend_->exec_op(op, inputs, output);
				});

				vector<tensor_p> outputs(cg_.out_nodes_.size());
				std::transform(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), 
						outputs.begin(), [this](auto id) { return tensors_[id]; });
			}
		private:
			const exec_params& params_;
			const call_graph& cg_;
			borrowed_ptr<backend_t> backend_;
			section_exec_tensors& tensors_;
	};

	/**
	 * A section is a component that is responsible for managing resources for operations 
	 * on a call graph, i.e. execution/differentials.
	 */
	class env_section {
		public:
			env_section(borrowed_ptr<exec_env> env, borrowed_ptr<backend_t> backend,
					const call_graph& cg,
					const hash_map<node_id, tensor_p>& data_tensors,
					section_resources&& resources,
					unique_ptr<diff_env>&& diff_env,
					unique_ptr<diff_info>&& diff_info
					):
				cg_{cg}, env_{env}, backend_{backend},
				data_tensors_{data_tensors}, 
				resources_{std::move(resources)},
				diff_env_{std::move(diff_env)},
				diff_info_{std::move(diff_info)} {}

			exec_result execute(exec_params& params) {
				//prepare run
				reset_internal_tensors();
				if (params.calc_diffs) {
					diff_env_->reset();
				}
				section_exec_tensors tensors{resources_.internal_tensors_, data_tensors_,
					params.inputs_, params.outputs_};
				
				//start run
				section_executor exec{params, cg_, backend_, tensors};
				exec.execute();

				exec_result result{.success_=true};

				//calculate derivatives if required
				if (params.calc_diffs) {
					diff_env_->calc_diffs(tensors);
					result.grad_system_ = diff_env_->get_grad_system();
				}

				return result;
			}

			tensor_p& get_tensor(node_id id) {
				if (data_tensors_.contains(id)) {
					return data_tensors_.at(id);
				} else {
					return resources_.internal_tensors_.at(id);
				}
			}
			

		private:
			void reset_internal_tensors() {
				for (auto& [id, tens]: resources_.internal_tensors_) {
					tens->back()->zero();
				}
			}

			const call_graph& cg_;
			unique_ptr<diff_info> diff_info_;
			unique_ptr<diff_env> diff_env_;
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;

			hash_map<node_id, tensor_p> data_tensors_;
			section_resources resources_;

			friend class EnvSection_Execute_Test;
	};

	class env_section_builder {
		public:
			env_section_builder(borrowed_ptr<exec_env> env, borrowed_ptr<backend_t> backend,
					const call_graph& cg) :  
				cg_{cg}, env_{env}, backend_{backend} {}

			env_section_builder& set_data_tensors(const hash_map<node_id, tensor_p>& data_tensors) {
				data_tensors_ = data_tensors;
				return *this;
			}

			env_section_builder& allocate_internal_tensors() {
				for (auto intn_id: cg_.internal_nodes_) {
					auto& node = cg_.flow_nodes_.at(intn_id);
					resources_.internal_tensors_[intn_id] = env_->create_tensor(node.shape_);
				}
				return *this;
			}

			env_section_builder& create_diff_info() {
				diff_info_builder builder{cg_};
				diff_info_ = builder
					.all_data_nodes()
					.find_dependencies()
					.build();
				return *this;
			}

			env_section_builder& create_fp_diff() {
				diff_env_ = std::make_unique<fp_diff_env>(cg_, diff_info_.get(), backend_);
				((fp_diff_env*)diff_env_.get())->init();
				return *this;
			}

			env_section_builder& create_bw_diff() {
				bw_diff_env_builder builder{cg_, diff_info_.get(), backend_};
				diff_env_ = builder.allocate_grad_tensors().build();
				return *this;
			}


			[[nodiscard]]
			unique_ptr<env_section> build() {
				return std::make_unique<env_section>(env_, backend_, cg_, data_tensors_, 
					std::move(resources_), std::move(diff_env_), std::move(diff_info_));
			}
		private:
			const call_graph& cg_;
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;
			hash_map<node_id, tensor_p> data_tensors_;
			unique_ptr<diff_info> diff_info_;
			unique_ptr<diff_env> diff_env_;

			section_resources resources_;

	};

}

