#pragma once

#include <bits/ranges_algo.h>
#include <cassert>
#include <rep/call_graph.h>

namespace plearn::rep {

	/**
	 *  Class that runs over the operations of a call graph,
	 *  in an order that respects the dependencies between the operations.
	 *  I.e. if an operation depends on the result of another operation,
	 *  the latter will be reached before the former.
	 */
	class call_graph_forward_runner {
		struct node_info {
				op_node_id id_;
				int deps_ = 0;
				int unready_deps_ = 0;
		};
		public:
			call_graph_forward_runner(const call_graph& cg) : cg_{cg} {
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

			template<typename Callable>
				void run(Callable&& op_action)
				requires std::invocable<Callable, const op_node&>
				{
					reset();
					while (state_ == run_state::IN_PROGRESS) {
						auto& opn_id = *ready_ops_.begin();
						auto& opn = cg_.op_nodes_.at(opn_id);
						op_action(opn);
						op_finished(opn_id);
					}
				}




			/**
			 * Start a run: reset dependency counters and available operations.
			 */
			void reset() {
				assert(state_ == run_state::READY);
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
				assert(state_ == run_state::IN_PROGRESS);
				//TODO concurrency
				//remove op from ready ops
				ready_ops_.erase(op);
				//decrement dependencies of output tensor
				auto out_tensn_id = cg_.op_nodes_.at(op).out_;
				for (auto op_id: cg_.flow_nodes_.at(out_tensn_id).outputs_) {
					decrement_deps(op_info_.at(op_id));
				}
				//check if op out tensor is a graph output
				if (std::ranges::count(cg_.out_nodes_, out_tensn_id) > 0) {
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


	class call_graph_backward_runner {
		public:
			call_graph_backward_runner(const call_graph& cg) : cg_{cg} { }

			template<typename Callable>
				void run(Callable&& op_action)
				requires std::invocable<Callable, const op_node&>
				{
					reset();
					while (state_ == run_state::IN_PROGRESS) {
						auto& opn_id = *ready_ops_.begin();
						auto& opn = cg_.op_nodes_.at(opn_id);
						op_action(opn);
						op_finished(opn_id);
					}
				}

			void reset() {
				assert(state_ == run_state::READY);
				state_ = run_state::IN_PROGRESS;
				unready_in_tens_ = cg_.in_nodes_.size();
				//clear ready ops and find initial ready ops
				ready_ops_ = unordered_set<op_node_id>{};
				for (auto outn_id: cg_.out_nodes_) {
					auto input = cg_.flow_nodes_.at(outn_id).input_;
					if (input.has_value()) 
						ready_ops_.insert(input.value());
				}
			}

			void op_finished(op_node_id opn_id) {
				assert(state_ == run_state::IN_PROGRESS);
				//TODO concurrency
				//remove op from ready ops
				ready_ops_.erase(opn_id);
				//add op inputs to ready ops
				for (auto inn_id: cg_.op_nodes_.at(opn_id).inputs_) {
					auto& inn = cg_.flow_nodes_.contains(inn_id) ?
						cg_.flow_nodes_.at(inn_id) : cg_.data_nodes_.at(inn_id);
					if (inn.input_.has_value()) {
						ready_ops_.insert(inn.input_.value());
					} else if (std::ranges::count(cg_.in_nodes_, inn_id) > 0) {
						//input is a graph input
						unready_in_tens_--;
						if (unready_in_tens_ == 0) {
							state_ = run_state::READY;
						}
					}
				}
			}

			run_state state() const { return state_; }

			const unordered_set<op_node_id>& ready_ops() const { return ready_ops_; }

		private:
			const call_graph& cg_;
			run_state state_ = run_state::READY;
			unordered_set<op_node_id> ready_ops_;
			int unready_in_tens_;
	};
}
