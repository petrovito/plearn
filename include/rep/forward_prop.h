#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <rep/call_graph.h>
#include <rep/call_graph_runner.h>


namespace plearn::rep {

	/**
	 * Describes derivatives of a tensor node.
	 * Both direct and indirect derivatives are stored.
	 *
	 * Direct derivatives are the ones that are computed using the op and the input nodes for the op.
	 *
	 * The indirect derivatives are WRT the variables of the system. They are computed using the direct derivatives
	 * and their indirect derivatives using the chain rule.
	 */
	struct op_derivative {
		op_derivative() = default;
		op_derivative(const op_node& op_node, const vector<node_id>& vars) :
			op_node_{op_node} {
				for	(auto varn_id: vars) {
					indirect_grads_[varn_id] = independent_gradient;
				}
			};
		
		static op_derivative identity(tensor_node_id tenn_id, const vector<node_id>& vars) {
			op_derivative d{op_node::identity(tenn_id), vars};
			d.direct_grads_[tenn_id] = identity_gradient;
			d.indirect_grads_[tenn_id] = identity_gradient;
			return d;
		}
		static op_derivative independent(tensor_node_id tenn_id, const vector<node_id>& vars) {
			op_node noopn{-1, noop{}, {}, tenn_id};
			op_derivative d{noopn, vars};
			return d;
		}

		/* const gradient& operator[](node_id id) const { */
		/* 	if (direct_grads_.contains(id)) return direct_grads_.at(id); */
		/* 	return independent_gradient; */
		/* } */

		bool depends_on(node_id id) const {
			/* if (!indirect_grads_.contains(id)) return false; */
			return !indirect_grads_.at(id).independent_;
		}

		bool depends_on_any() const { //any variable node, that is
			for (auto& [id, grad]: indirect_grads_) {
				if (!grad.independent_) return true;
			}
			return false;
		}

		unordered_set<node_id> variable_dependencies() const {
			unordered_set<node_id> deps{};
			for (auto& [id, grad]: indirect_grads_) {
				if (!grad.independent_) deps.insert(id);
			}
			return deps;
		}

		bool has_non_trivial_grads() {
			for (auto& [id, grad]: indirect_grads_) {
				if (!grad.identity_ && !grad.independent_)
					return true;
			}
			return false;
		}

		op_node op_node_;//TODO should be reference, except identity is problematic
		hash_map<node_id, dep_type> direct_grads_{}; //calculated from op and input tensor values
		hash_map<node_id, dep_type> indirect_grads_{}; //calculated from direct_grads_
	};


	class diff_info {

		public:
			diff_info(
					vector<node_id>&& variable_nodes,
					vector<node_id>&& output_nodes,
					hash_map<node_id, op_derivative>&& derivatives
					) :
				variable_nodes_{std::move(variable_nodes)},
				output_nodes_{std::move(output_nodes)},
				derivatives_{std::move(derivatives)} {} ;

			//getters
			const vector<node_id>& variable_nodes() const { return variable_nodes_; }
			const hash_map<node_id, op_derivative>& derivatives() const { return derivatives_; }
		private:
			vector<node_id> variable_nodes_;
			vector<node_id> output_nodes_;
			hash_map<node_id, op_derivative> derivatives_;
	};
	

	class diff_info_builder {

		public:
			diff_info_builder(const call_graph& graph) :
				graph_{graph} { }
			
			/**
			 * The variable tensors of the system.
			 * I.e. calculate the derivatives WRT these tensors.
			 */
			diff_info_builder& variable_nodes(const vector<node_id>& data_nodes) {
				variable_nodes_ = data_nodes;
				return *this;
			}

			/**
			 * Set all data nodes to be variable nodes.
			 */
			diff_info_builder& all_data_nodes() {
				for (auto& [id, node]: graph_.data_nodes_) {
					variable_nodes_.push_back(id);
				}
				return *this;
			}

			/**
			 * Find the nodes that are dependant on the requested data nodes.
			 */
			diff_info_builder& find_depending_tensors() {
				if (variable_nodes_.empty()) {
					//set variables to be all data nodes
					all_data_nodes();
				}
				//add identity derivatives for the data tensors
				for (auto var_node_id: variable_nodes_) {
					derivatives_[var_node_id] =
							op_derivative::identity(var_node_id, variable_nodes_);
				}
				//set independent derivatives for all input nodes
				for (auto inn_id: graph_.in_nodes_) {
					derivatives_[inn_id] = op_derivative::independent(inn_id, variable_nodes_);
				}
				//find derivatives for all flow node-tensors
				call_graph_forward_runner runner{graph_};
				runner.run([this] (const op_node& opn) {
					//iterate over inputs and find dependant variable tensors for output recursively
					op_derivative out_diffs{opn, variable_nodes_};
					for (auto inn_id: opn.inputs_) {
						out_diffs.direct_grads_[inn_id] = {};
						auto& in_diffs = derivatives_.at(inn_id);
						//set dependencies
						for (auto varn_id: variable_nodes_) {
							if (in_diffs.depends_on(varn_id)) {
								out_diffs.indirect_grads_[varn_id].independent_ = false;
							}
						}
					}
					//add op diff and mark op as finished
					derivatives_[opn.out_] = out_diffs;
				});
				return *this;
			}

			unique_ptr<diff_info> build() {
				return make_unique<diff_info>(
						std::move(variable_nodes_),
						std::move(output_nodes_),
						std::move(derivatives_)
						);
			}


		private:
			const call_graph& graph_;
			vector<node_id> variable_nodes_;
			vector<node_id> output_nodes_;
			hash_map<node_id, op_derivative> derivatives_;
	};
	
}

