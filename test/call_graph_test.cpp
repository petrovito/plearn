#include <gtest/gtest.h>
#include <call_graph.h>

using namespace plearn;

TEST(CallGraph, BasicCall) {
	
	tensor in;
	tensor int1, int2;
	tensor temporary;
	tensor out;

	placeholder_node in_node{.shape_={1, {768}}},
					 temp{.shape_={1, {32}}},
					 out_node{.shape_={1,{1}}};
	tensor_node      t_node1{},
					 t_node2{};

	matmul mul_op1,
		   mul_op2;

	op_node op_node1{.op_{mul_op1}},
			op_node2{.op_{mul_op2}};

	call_graph graph{.in_nodes_{{0, in_node}}, .int_nodes_{{1,temp}}, 
		.out_nodes_{{2, out_node}},
					 .data_nodes_{{3,t_node1}, {4,t_node2}}, 
					 .op_nodes_{{5, op_node1}, {6, op_node2}}};

	


}

