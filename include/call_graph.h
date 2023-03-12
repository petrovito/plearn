#pragma once

#include <cstdint>
#include <operation.h>
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
	};

	class tensor {
		public:
			tensor() = default;
			tensor(const shape& s) : shape_{s} {}
			shape shape_;

	};

	enum class op_type {
		matmul,
		matvecmul,
		add,
	};

	struct operation {
		op_type type_;
	};

	struct matmul : public operation {

	};



	using node_id = int;

	struct op_node;

	struct tensor_node {
		node_id id_;
		shape shape_;
		vector<node_id> outputs_;
		tensor* tensor_;
	};

	struct op_node {
		node_id id_;
		operation op_;

		vector<node_id> inputs_;
		node_id out_;
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
	};
	
	class node_graph_builder {

		call_graph build() { return {}; }

	};

}
