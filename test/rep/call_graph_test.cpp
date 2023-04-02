#include <gtest/gtest.h>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/call_graph_runner.h>

using namespace plearn::rep;

TEST(CallGraph, Builder) {

	call_graph_builder builder;

	auto inn_id = builder.add_input_node(shape_t{768});
	auto data1n_id = builder.add_data_node(shape_t{768, 128});
	auto [op1n_id, flown_id] = 
		builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape_t{128});
	auto data2n_id = builder.add_data_node(shape_t{128, 10});
	auto [op2n_id, outn_id] = 
		builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape_t{10});
	builder.make_output(outn_id);

	auto cg = builder.build();

	ASSERT_EQ(cg.in_nodes_.size(), 1);
	ASSERT_EQ(cg.in_nodes_[0], inn_id);
	ASSERT_EQ(cg.out_nodes_.size(), 1);
	ASSERT_EQ(cg.out_nodes_[0], outn_id);
	ASSERT_EQ(cg.internal_nodes_.size(), 1);
	ASSERT_EQ(cg.internal_nodes_[0], flown_id);
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

TEST(CallGraph, Runner) {
	call_graph_builder builder;

	auto inn_id = builder.add_input_node(shape_t{768});
	auto data1n_id = builder.add_data_node(shape_t{768, 128});
	auto [op1n_id, flown_id] = 
		builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape_t{128});
	auto data2n_id = builder.add_data_node(shape_t{128, 10});
	auto [op2n_id, outn_id] = 
		builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape_t{10});
	builder.make_output(outn_id);

	auto cg = builder.build();

	auto fw_runner = call_graph_forward_runner(cg);
	{
	fw_runner.reset();
	auto& ready_ops = fw_runner.ready_ops();
	ASSERT_EQ(ready_ops.size(), 1);
	ASSERT_TRUE(ready_ops.contains(op1n_id));
	fw_runner.op_finished(op1n_id);
	ASSERT_EQ(ready_ops.size(), 1);
	ASSERT_TRUE(ready_ops.contains(op2n_id));
	fw_runner.op_finished(op2n_id);
	ASSERT_EQ(ready_ops.size(), 0);
	ASSERT_EQ(fw_runner.state(), run_state::READY);
	}

	auto bw_runner = call_graph_backward_runner(cg);
	{
	bw_runner.reset();
	auto& ready_ops = bw_runner.ready_ops();
	ASSERT_EQ(ready_ops.size(), 1);
	ASSERT_TRUE(ready_ops.contains(op2n_id));
	bw_runner.op_finished(op2n_id);
	ASSERT_EQ(ready_ops.size(), 1);
	ASSERT_TRUE(ready_ops.contains(op1n_id));
	bw_runner.op_finished(op1n_id);
	ASSERT_EQ(ready_ops.size(), 0);
	ASSERT_EQ(bw_runner.state(), run_state::READY);
	}

}

