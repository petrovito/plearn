#pragma once

#include "rep/call_graph.h"

namespace plearn::rep {

	/**
	 *  Class that runs over the operations of a call graph,
	 *  in an order that respects the dependencies between the operations.
	 *  I.e. if an operation depends on the result of another operation,
	 *  the latter will be reached before the former.
	 */
	class call_graph_runner {
		struct node_info {
				op_node_id id_;
				int deps_ = 0;
				int unready_deps_ = 0;
		};
		public:
			call_graph_runner(const call_graph& cg) : cg_{cg} {
				//add node_info for each op node
				for (auto& [id, op]: cg_.op_nodes_) {
					op_info_[id] = {id};
				}
				//count flow dependencies
				for (auto& [id, flow_n]: cg_.flow_nodes_) {
					for (auto op_id: flow_n.outputs_) {
						op_info_[op_id].deps_++;
					}
				}
			}

			/**
			 * Start a run: reset dependency counters and available operations.
			 */
			void reset() {
				state_ = run_state::IN_PROGRESS;
				unready_out_tens_ = cg_.out_nodes_.size();
				//reset dependency counters
				for (auto& [id, op]: cg_.op_nodes_) {
					op_info_[id].unready_deps_ = op_info_[id].deps_;
				}
				//clear ready ops and find initial ready ops
				ready_ops_ = unordered_set<op_node_id>{};
				for (auto inn_id: cg_.in_nodes_) {
					for (auto op_id: cg_.flow_nodes_.at(inn_id).outputs_) {
						decrement_deps(op_info_.at(op_id));
					}
				}
			}

			/**
			 * Call this function when an operation has finished executing.
			 * Updates dependency counters and available operations.
			 */
			void op_finished(op_node_id op) {
				//TODO concurrency
				//remove op from ready ops
				ready_ops_.erase(op);
				//decrement dependencies of output tensor
				auto out_tensn_id = cg_.op_nodes_.at(op).out_;
				for (auto op_id: cg_.flow_nodes_.at(out_tensn_id).outputs_) {
					decrement_deps(op_info_.at(op_id));
				}
				//check if op out tensor is a graph output
				if (std::count(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), out_tensn_id) > 0) {
					unready_out_tens_--;
					if (unready_out_tens_ == 0) {
						state_ = run_state::READY;
					}
				}
			}

			run_state state() const { return state_; }

			const unordered_set<op_node_id>& ready_ops() const { return ready_ops_; }

		private:
			void decrement_deps(node_info& n_info) {
				n_info.unready_deps_--;
				if (n_info.unready_deps_ == 0) {
					ready_ops_.insert(n_info.id_);
				}
			}

			const call_graph& cg_;
			run_state state_ = run_state::READY;
			unordered_set<op_node_id> ready_ops_;
			hash_map<op_node_id, node_info> op_info_;
			int unready_out_tens_;
			
	};



}
