#include <gtest/gtest.h>
#include <call_graph.h>

using namespace plearn;

TEST(CallGraph, Builder) {

	call_graph_builder builder;

	auto inn_id = builder.add_input_node(shape{768});
	auto data1n_id = builder.add_data_node(shape{768, 128});
	auto [op1n_id, flown_id] = 
		builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape{128});
	auto data2n_id = builder.add_data_node(shape{128, 10});
	auto [op2n_id, outn_id] = 
		builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape{10});
	builder.make_output(outn_id);

	auto cg = builder.build();

	ASSERT_EQ(cg.in_nodes_.size(), 1);
	ASSERT_EQ(cg.in_nodes_[0], inn_id);
	ASSERT_EQ(cg.out_nodes_.size(), 1);
	ASSERT_EQ(cg.out_nodes_[0], outn_id);
	ASSERT_EQ(cg.flow_nodes_.size(), 3);
	ASSERT_EQ(cg.data_nodes_.size(), 2);
	ASSERT_EQ(cg.op_nodes_.size(), 2);
	
	ASSERT_EQ(cg.flow_nodes_[inn_id].outputs_.size(), 1);
	ASSERT_EQ(cg.flow_nodes_[inn_id].outputs_[0], op1n_id);
	ASSERT_EQ(cg.data_nodes_[data1n_id].outputs_.size(), 1);
	ASSERT_EQ(cg.data_nodes_[data1n_id].outputs_[0], op1n_id);
	ASSERT_EQ(cg.op_nodes_[op1n_id].inputs_.size(), 2);
	ASSERT_EQ(cg.op_nodes_[op1n_id].inputs_[0], inn_id);
	ASSERT_EQ(cg.op_nodes_[op1n_id].inputs_[1], data1n_id);
	ASSERT_EQ(cg.op_nodes_[op1n_id].out_, flown_id);

	ASSERT_EQ(cg.flow_nodes_[flown_id].outputs_.size(), 1);
	ASSERT_EQ(cg.flow_nodes_[flown_id].outputs_[0], op2n_id);
	ASSERT_EQ(cg.data_nodes_[data2n_id].outputs_.size(), 1);
	ASSERT_EQ(cg.data_nodes_[data2n_id].outputs_[0], op2n_id);
	ASSERT_EQ(cg.op_nodes_[op2n_id].inputs_.size(), 2);
	ASSERT_EQ(cg.op_nodes_[op2n_id].inputs_[0], flown_id);
	ASSERT_EQ(cg.op_nodes_[op2n_id].inputs_[1], data2n_id);
	ASSERT_EQ(cg.op_nodes_[op2n_id].out_, outn_id);
}

