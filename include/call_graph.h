#pragma once

#include <bits/iterator_concepts.h>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include "operation.h"
#include <tuple>
#include <unordered_map>
#include <vector>

namespace plearn {

	using std::vector;

	struct shape {
		int rank;
		vector<std::uint64_t> dims;

		uint64_t size() const {
			std::size_t size = 1;
			for (auto dim : dims) {
				size *= dim;
			}
			return size;
		}
		shape() = default;

		/* template<typename IntType, template <typename> class coll> */
		/* shape(const coll<IntType>& _dims) : */ 
		/* 	rank{_dims.size()}, dims{} { */
		/* 		std::copy(_dims.begin(), _dims.end(), dims.begin()); */
		/* 	} */

		shape(vector<uint64_t> _dims) : 
			rank{_dims.size()}, dims{_dims} {}

		shape(std::integral auto...dims) : 
			rank{sizeof...(dims)}, dims{dims...} {}

		friend auto operator<=>(const shape&, const shape&) = default;
	};

	class tensor {
		public:
			tensor() = default;
			tensor(const shape& s) : shape_{s} {}
			shape shape_;

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

	struct tensor_node {
		node_id id_;
		shape shape_;
		vector<node_id> outputs_{};

		friend auto operator<=>(const tensor_node&, const tensor_node&) = default;
	};

	struct op_node {
		node_id id_;
		operation op_;

		vector<node_id> inputs_;
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
			node_id add_input_node(shape s) {
				return add_flow_node(s, true);
			}

			node_id add_data_node(shape s) {
				node_id id = next_id_++;
				tensor_node tn{id, s};
				data_nodes_[id] = tn;
				tensor_nodes_[id] = &data_nodes_[id];
				return id;
			}
			
			//return
			std::tuple<node_id, node_id> add_op_node(operation op,
					vector<node_id> inputs, shape out_shape) {
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
			node_id add_flow_node(shape s, bool is_input=false) {
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

}
