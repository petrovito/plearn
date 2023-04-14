#pragma once

#include <bits/ranges_algo.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/call_graph_runner.h>
#include <algorithm>
#include <rep/diff_info.h>
#include <environ/env_types.h>
#include <environ/exec_env.h>
#include <environ/env_page.h>
#include <environ/fp_diff_env.h>
#include "environ/bw_diff_page.h"

namespace plearn::env {

	/**
	 * A section is a component that is responsible for managing resources for operations 
	 * on a call graph, i.e. execution/differentials.
	 * Contains env_pages.
	 */
	class env_section {
		public:
			env_section(
					const call_graph& cg, unique_ptr<diff_info>&& diff_info,
				    borrowed_ptr<exec_env> env, borrowed_ptr<backend_t> backend,
					const unordered_map<node_id, tensor_p>& data_tensors
					):
				cg_{cg}, diff_info_{std::move(diff_info)},
				env_{env}, backend_{backend},
				data_tensors_{data_tensors} 
			{}

			exec_result execute(exec_params& params) {
				ensure_resources(params);
				return env_page_->execute(params);
			}

			tensor_p& get_data_tensor(node_id id) {
				if (data_tensors_.contains(id)) {
					return data_tensors_.at(id);
				}
				throw std::runtime_error("tensor not found");
			}

			void set_data_tensor(node_id id, tensor_p tens) {
				data_tensors_[id] = tens;
			}

			void create_env_page(bool make_diff_page = false) {
				unique_ptr<exec_page> exec_page = create_exec_page();
				env_page_ = std::make_unique<env_page>(
					std::move(exec_page), data_tensors_);

				if (make_diff_page) {
					unique_ptr<diff_page> diff_page = create_diff_page();
					env_page_->set_diff_page(std::move(diff_page));
				}
			}

			

		private:
			void ensure_resources(exec_params& params) {
				if (!env_page_.get()) {
					create_env_page(params.calc_diffs);
				}
				if (params.calc_diffs) {
					if (!env_page_->has_diff_page())
						env_page_->set_diff_page(create_diff_page());
				}
			}

			//create exec page and allocate memory
			unique_ptr<exec_page> create_exec_page() {
				exec_page_resources resources;
				for (auto intn_id: cg_.internal_nodes_) {
					auto& node = cg_.flow_nodes_.at(intn_id);
					resources.internal_tensors_[intn_id] = env_->create_tensor(node.shape_);
				}
				return std::make_unique<exec_page>(backend_, cg_, std::move(resources));
			}

			//create diff page and allocate memory
			unique_ptr<diff_page> create_diff_page() {
				bw_diff_page_builder builder{cg_, diff_info_.get(), backend_};
				return builder.allocate_grad_tensors().build();
			}


			//representations
			const call_graph& cg_;
			unique_ptr<diff_info> diff_info_;

			//resources managed by this section
			unordered_map<node_id, tensor_p> data_tensors_;

			//subcomponents
			unique_ptr<env_page> env_page_;

			//calculation and resource mgmt components
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;

			friend class EnvSection_Execute_Test;
	};

	class env_section_builder {
		public:
			env_section_builder(borrowed_ptr<exec_env> env, borrowed_ptr<backend_t> backend,
					const call_graph& cg) :  
				cg_{cg}, env_{env}, backend_{backend} {}

			env_section_builder& set_data_tensors(unordered_map<node_id, tensor_p>&& data_tensors) {
				data_tensors_ = data_tensors;
				return *this;
			}

			// env_section_builder& create_fp_diff() {
			// 	diff_env_ = std::make_unique<fp_diff_env>(cg_, diff_info_.get(), backend_);
			// 	((fp_diff_env*)diff_env_.get())->init();
			// 	return *this;
			// }


			[[nodiscard]]
			unique_ptr<env_section> build() {
				create_diff_info();
				return std::make_unique<env_section>(
						cg_, std::move(diff_info_),
						env_, backend_, 
						data_tensors_);
			}
		private:
			void create_diff_info() {
				diff_info_builder builder{cg_};
				diff_info_ = builder
					.all_data_nodes()
					.find_dependencies()
					.build();
			}

			const call_graph& cg_;
			borrowed_ptr<exec_env> env_;
			borrowed_ptr<backend_t> backend_;
			unordered_map<node_id, tensor_p> data_tensors_;
			unique_ptr<diff_info> diff_info_;
			unique_ptr<diff_page> diff_env_;

			exec_page_resources resources_;

	};

}

