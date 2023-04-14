#pragma once

#include "rep/call_graph_runner.h"
#include <environ/env_types.h>
#include <unordered_map>

namespace plearn::env {
	/**
	 * Holds resourcers for ONE execution of a call graph.
	 * Resources are managed by another component.
	 *
	 * Resources means internal tensors, that are not provided by an external component.
	 */
	struct exec_page_resources {
		unordered_map<node_id, tensor_p> internal_tensors_;
	};

	class diff_page {
		public:
			virtual void calc_diffs(exec_page_tensors&) = 0;
			virtual void reset() = 0;
			virtual borrowed_ptr<grad_system> get_grad_system() = 0;
			virtual ~diff_page() = default;
	};

	class exec_page {
		public:
            exec_page(borrowed_ptr<backend_t> backend, const call_graph& cg, 
                      exec_page_resources&& resources):
                cg_{cg}, resources_{std::move(resources)}, backend_{backend} {}

			exec_result execute(exec_page_tensors& tensors) {
				reset();
				call_graph_forward_runner runner{cg_};
                runner.run([&](const op_node& opn) {
				
					auto& op = opn.op_;
					//collect inputs and outputs
					vector<tensor_p> inputs(opn.inputs_.size());
					std::transform(opn.inputs_.begin(), opn.inputs_.end(), 
							inputs.begin(), 
                            [this, &tensors](auto id) { return tensors[id]; });
					tensor_p output = tensors[opn.out_];
					//execute operation
					backend_->exec_op(op, inputs, output);
				});

				vector<tensor_p> outputs(cg_.out_nodes_.size());
				std::transform(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), 
						outputs.begin(), 
                        [this, &tensors](auto id) { return tensors[id]; });
				

				exec_result result{.success_=true};
				return result;
			}

            void reset() {
              for (auto &[id, tens] : resources_.internal_tensors_) {
                tens->back()->zero();
              }
            }

            exec_page_resources& resources() { return resources_; }
		private:
            //representation
			const call_graph& cg_;

			//resources held by this page
			exec_page_resources resources_;

			//calculation and resource mgmt components
			borrowed_ptr<backend_t> backend_;
	};

    /**
      * A pair of exec_page and diff_page.
    */
	class env_page {
        public:
			env_page(
					// const call_graph& cg, 
			   		 unique_ptr<exec_page>&& exec_page,
					 unordered_map<node_id, tensor_p>& data_tensors):
				// cg_{cg}, 
				exec_page_{std::move(exec_page)}, data_tensors_{data_tensors} {}

            exec_result execute(exec_params& params) {
				exec_page_tensors exec_tensors{
                    exec_page_->resources().internal_tensors_,
                    data_tensors_, 
					params.inputs_, params.outputs_};


                auto result = exec_page_->execute(exec_tensors);

				if (params.calc_diffs) {
                    diff_page_->calc_diffs(exec_tensors);
					result.grad_system_ = diff_page_->get_grad_system();
				}
                return result;
            }
			
			exec_result batch_execute(exec_params& params) {
				exec_page_tensors exec_tensors{
					exec_page_->resources().internal_tensors_,
					data_tensors_, 
					params.inputs_, params.outputs_};

				auto result = exec_page_->execute(exec_tensors);

				if (params.calc_diffs) {
					diff_page_->calc_diffs(exec_tensors);
					result.grad_system_ = diff_page_->get_grad_system();
				}
				return result;
			}

			void set_diff_page(unique_ptr<diff_page>&& diff_page) {
				diff_page_ = std::move(diff_page);
			}

			bool has_diff_page() const { return diff_page_ != nullptr; }

		private:
			//representation
			// const call_graph& cg_;

			//subcomponents
			unique_ptr<exec_page> exec_page_;
			unique_ptr<diff_page> diff_page_;

			//resources managed by other components
			unordered_map<node_id, tensor_p>& data_tensors_;
	};
}
