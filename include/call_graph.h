#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include "operation.h"
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace plearn {

	using std::vector;

	struct shape_t {
		int rank;
		vector<std::uint64_t> dims;

		uint64_t size() const {
			std::size_t size = 1;
			for (auto dim : dims) {
				size *= dim;
			}
			return size;
		}
		shape_t() = default;

		/* template<typename IntType, template <typename> class coll> */
		/* shape(const coll<IntType>& _dims) : */ 
		/* 	rank{_dims.size()}, dims{} { */
		/* 		std::copy(_dims.begin(), _dims.end(), dims.begin()); */
		/* 	} */

		shape_t(vector<uint64_t> _dims) : 
			rank{_dims.size()}, dims{_dims} {}

		shape_t(std::integral auto...dims) : 
			rank{sizeof...(dims)}, dims{dims...} {}

		friend auto operator<=>(const shape_t&, const shape_t&) = default;
	};

	using tensor_id = uint64_t;

	class tensor {
		public:
			tensor() = default;
			shape_t shape() const { return shape_; }
			tensor_id id() const { return id_; }
		private:
			tensor(const shape_t& s, tensor_id id) : shape_{s}, id_{id} {}
			shape_t shape_;
			tensor_id id_;
		friend class tensor_factory;

	};
	
	class tensor_factory {
		public:
			static tensor create(const shape_t& s) {
				return tensor{s, next_id()};
			}
		private:
			static tensor_id next_id() {
				static tensor_id id = 1;
				return id++;
			}
	};

	enum class op_type {
		noop,
		matmul,
		matvecmul,
		add,
	};

	struct operation {
		op_type type_;
		
		friend auto operator<=>(const operation&, const operation&) = default;
	};

	struct noop : public operation {
		noop() : operation{op_type::noop} { }
	};

	struct matmul : public operation {
		matmul() : operation{op_type::matmul} {}
	};

	struct matvecmul : public operation {
		matvecmul() : operation{op_type::matvecmul} {}
	};



	using node_id = int;
	using op_node_id = node_id;

	struct tensor_node {
		node_id id_;
		shape_t shape_;
		vector<node_id> outputs_{};

		friend auto operator<=>(const tensor_node&, const tensor_node&) = default;
	};

	struct op_node {
		node_id id_;
		operation op_;

		vector<node_id> inputs_; //dependencies
		node_id out_{};

		friend auto operator<=>(const op_node&, const op_node&) = default;
	};


	template<typename K, typename V>
	using hash_map = std::unordered_map<K, V>;

	class call_graph {
		public:
			hash_map<node_id, tensor_node> flow_nodes_;
			hash_map<node_id, tensor_node> data_nodes_;
			hash_map<node_id, op_node> op_nodes_;

			vector<node_id> in_nodes_;
			vector<node_id> out_nodes_;

			friend bool operator==(const call_graph& a, const call_graph& b) = default;	
	};

	class call_graph_builder {
		public:
			node_id add_input_node(shape_t s) {
				return add_flow_node(s, true);
			}

			node_id add_data_node(shape_t s) {
				node_id id = next_id_++;
				tensor_node tn{id, s};
				data_nodes_[id] = tn;
				tensor_nodes_[id] = &data_nodes_[id];
				return id;
			}
			
			//return
			std::tuple<node_id, node_id> add_op_node(operation op,
					vector<node_id> inputs, shape_t out_shape) {
				node_id id = next_id_++;
				auto out_id = add_flow_node(out_shape);
				op_node on{id, op, inputs, out_id};
				op_nodes_[id] = on;
				//add op to tensor node outputs
				for (auto node_id: inputs) {
					tensor_nodes_[node_id]->outputs_.push_back(id);
				}
				return {id, out_id};
			}

			void make_output(node_id id) {
				out_nodes_.push_back(id);
			}

			call_graph build() {
				return {
					flow_nodes_,
					data_nodes_,
					op_nodes_,
					in_nodes_,
					out_nodes_,
				};
			}

		private:
			node_id add_flow_node(shape_t s, bool is_input=false) {
				node_id id = next_id_++;
				tensor_node tn{id, s};
				flow_nodes_[id] = tn;
				tensor_nodes_[id] = &flow_nodes_[id];
				if (is_input)
					in_nodes_.push_back(id);
				return id;
			}

			hash_map<node_id, tensor_node*> tensor_nodes_;
			hash_map<node_id, tensor_node> flow_nodes_;
			hash_map<node_id, tensor_node> data_nodes_;
			hash_map<node_id, op_node> op_nodes_;
			std::vector<node_id> in_nodes_;
			std::vector<node_id> out_nodes_;
			node_id next_id_ = 0;

	};

	enum class env_state {
		IN_PROGRESS,
		READY,
	};

	using std::unordered_set;

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
				state_ = env_state::IN_PROGRESS;
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
				//check if output tensor is ready
				if (std::count(cg_.out_nodes_.begin(), cg_.out_nodes_.end(), out_tensn_id) > 0) {
					unready_out_tens_--;
					if (unready_out_tens_ == 0) {
						state_ = env_state::READY;
					}
				}
			}

			const unordered_set<op_node_id>& ready_ops() const { return ready_ops_; }

		private:
			void decrement_deps(node_info& n_info) {
				n_info.unready_deps_--;
				if (n_info.unready_deps_ == 0) {
					ready_ops_.insert(n_info.id_);
				}
			}

			const call_graph& cg_;
			env_state state_;
			unordered_set<op_node_id> ready_ops_;
			hash_map<op_node_id, node_info> op_info_;
			int unready_out_tens_;
			
	};

}
