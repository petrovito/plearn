#pragma once

#include "rep/rep_types.h"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include <ranges>
#include <unordered_set>
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
	struct node_diff_info {
		node_diff_info() = default;
		node_diff_info(const op_node& op_node, const vector<node_id>& vars) :
			op_node_{op_node} {
				for	(auto varn_id: vars) {
					variable_deps_[varn_id] = independent_gradient;
				}
			};
		
		static node_diff_info identity(tensor_node_id tenn_id, const vector<node_id>& vars) {
			node_diff_info d{op_node::identity(tenn_id), vars};
			d.op_input_deps_[tenn_id] = identity_gradient;
			d.variable_deps_[tenn_id] = identity_gradient;
			return d;
		}
		static node_diff_info independent(tensor_node_id tenn_id, const vector<node_id>& vars) {
			op_node noopn{-1, noop{}, {}, tenn_id};
			node_diff_info d{noopn, vars};
			return d;
		}

		bool depends_on(node_id id) const {
			return !variable_deps_.at(id).independent_;
		}

		bool depends_on_any() const { //any variable node, that is
			for (auto& [id, grad]: variable_deps_) {
				if (!grad.independent_) return true;
			}
			return false;
		}

		unordered_set<node_id> variable_dependencies() const {
			unordered_set<node_id> deps{};
			for (auto& [id, grad]: variable_deps_) {
				if (!grad.independent_) deps.insert(id);
			}
			return deps;
		}

		bool has_non_trivial_deps() {
			for (auto& [id, grad]: variable_deps_) {
				if (!grad.identity_ && !grad.independent_)
					return true;
			}
			return false;
		}

		bool output_dependant(node_id outn_id) {
			return !output_deps_.at(outn_id).independent_;
		}

		bool any_output_dependant() {
			for (auto& [id, grad]: output_deps_) {
				if (!grad.independent_)
					return true;
			}
			return false;
		}

		unordered_set<node_id> dependant_output_nodes() {
			unordered_set<node_id> deps{};
			for (auto& [id, grad]: output_deps_) {
				if (!grad.independent_) deps.insert(id);
			}
			return deps;
		}

		op_node op_node_;//TODO should be reference, except identity is problematic
		hash_map<node_id, dep_type> op_input_deps_{}; //describes dependencies on input tensors of the op
		hash_map<node_id, dep_type> variable_deps_{}; //describes dependencies on variables of the system
		hash_map<node_id, dep_type> output_deps_{}; //describes output dependencies WRT this node
	};


	class diff_info {

		public:
			diff_info(
					vector<node_id>&& variable_nodes,
					vector<node_id>&& output_nodes,
					hash_map<node_id, node_diff_info>&& derivatives
					) :
				variable_nodes_{std::move(variable_nodes)},
				output_nodes_{std::move(output_nodes)},
				dependencies_{std::move(derivatives)} {} ;

			//getters
			const vector<node_id>& variable_nodes() const { return variable_nodes_; }
			const hash_map<node_id, node_diff_info>& dependencies() const { return dependencies_; }
		private:
			vector<node_id> variable_nodes_;
			vector<node_id> output_nodes_;
			hash_map<node_id, node_diff_info> dependencies_;
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
			diff_info_builder& find_dependencies() {
				if (variable_nodes_.empty()) {
					//set variables to be all data nodes
					all_data_nodes();
				}
				//add identity derivatives for the data tensors
				for (auto var_node_id: variable_nodes_) {
					dependencies_[var_node_id] =
							node_diff_info::identity(var_node_id, variable_nodes_);
				}
				//set independent derivatives for all input nodes
				for (auto inn_id: graph_.in_nodes_) {
					dependencies_[inn_id] = node_diff_info::independent(inn_id, variable_nodes_);
				}
				//find derivatives for all flow node-tensors
				call_graph_forward_runner runner{graph_};
				runner.run([this] (const op_node& opn) {
					//iterate over inputs and find dependant variable tensors for output recursively
					node_diff_info out_diffs{opn, variable_nodes_};
					for (auto inn_id: opn.inputs_) {
						out_diffs.op_input_deps_[inn_id] = {};
						auto& in_diffs = dependencies_.at(inn_id);
						//set dependencies
						for (auto varn_id: variable_nodes_) {
							if (in_diffs.depends_on(varn_id)) {
								out_diffs.variable_deps_[varn_id].independent_ = false;
							}
						}
					}
					dependencies_[opn.out_] = out_diffs;
				});
				//set flow nodes against output nodes dependencoes as identity/independent
				for (auto& [flown_id, _]: graph_.flow_nodes_) {
					for (auto outn_id: graph_.out_nodes_) {
						if (flown_id != outn_id)
							dependencies_[flown_id].output_deps_[outn_id] = independent_gradient;
						else
							dependencies_[flown_id].output_deps_[outn_id] = identity_gradient;
					}
				}
				//set data nodes to be independent as well
				for (auto& [datan_id, _]: graph_.data_nodes_) {
					for (auto outn_id: graph_.out_nodes_) {
						dependencies_[datan_id].output_deps_[outn_id] = independent_gradient;
					}
				}
				//set output dependencies iteratively backwards
				call_graph_backward_runner b_runner{graph_};
				b_runner.run([this] (const op_node& opn) {
					auto& out_deps = dependencies_[opn.out_];
					for (auto inn_id: opn.inputs_) {
						auto& in_deps = dependencies_[inn_id];
						for (auto outn_id: graph_.out_nodes_) {
							if (out_deps.output_dependant(outn_id)) {
								in_deps.output_deps_[outn_id].independent_ = false;
							}
						}
					}
				});
				return *this;
			}

			unique_ptr<diff_info> build() {
				return make_unique<diff_info>(
						std::move(variable_nodes_),
						std::move(output_nodes_),
						std::move(dependencies_)
						);
			}


		private:
			const call_graph& graph_;
			vector<node_id> variable_nodes_;
			vector<node_id> output_nodes_;
			hash_map<node_id, node_diff_info> dependencies_;
	};
	
}

