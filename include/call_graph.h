#pragma once

#include <cstdint>
#include <operation.h>
#include <vector>

namespace plearn {

	using std::vector;

	struct shape {
		int rank;
		vector<std::uint64_t> dims;
	};

	class tensor {
		public:
			tensor() = default;
			tensor(shape s) : shape_{s} {}
			shape shape_;

	};

	struct operation {

	};

	struct matmul : public operation {

	};



	struct op_node;

	struct tensor_node {
		tensor* tensor_;

		/* vector<op_node*> inputs; */
		/* vector<op_node*> outputs; */
	};

	struct placeholder_node {
		shape shape_;
		vector<op_node*> outputs_;
	};

	struct op_node {
		operation op_;

		vector<tensor_node*> ints_;
		vector<placeholder_node*> deps_;
		/* vector<tensor_node*> out_; */
		placeholder_node* out_;
	};


			/* op_node& push_op_node(auto in_nodes, operation op) { */
			/* 	op_nodes_.push_back({.op_=op, .ins_=in_nodes}); */
			/* 	return op_nodes_.back(); */
			/* } */

			/* call_graph() {} */

		/* private: */

	class call_graph {

		public:

			vector<placeholder_node> in_nodes_;
			vector<placeholder_node> int_nodes_;
			vector<placeholder_node> out_nodes_;
			vector<tensor_node> data_nodes_;
			vector<op_node> op_nodes_;

			

	};

}
