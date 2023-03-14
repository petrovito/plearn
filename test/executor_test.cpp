#include "call_graph_test_helper.h"
#include "cpu_types.h"
#include "executor.h"
#include <gtest/gtest.h>
#include <call_graph.h>

namespace plearn {


TEST(CpuExecutor, ExecEnvBuilder) {

	call_graph_builder cg_builder;
	auto inn_id = cg_builder.add_input_node(shape{768});
	auto data1n_id = cg_builder.add_data_node(shape{768, 128});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape{128});
	auto data2n_id = cg_builder.add_data_node(shape{128, 10});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape{10});
	cg_builder.make_output(outn_id);
	auto cg = cg_builder.build();


	hash_map<node_id, cpu_tensor> data_tensors;
	for (auto& [id, node]: cg.data_nodes_) {
		data_tensors[id] = cpu_tensor_factory::allocate(node.shape_);
	}

	cpu_exec_env_builder builder(cg);
	auto env = builder.alloc_flow_mem()
		.load_data_nodes(data_tensors)
		.build();

	ASSERT_EQ(env->state_, env_state::READY);

	ASSERT_EQ(env->flow_nodes_.size(), 3);
	ASSERT_EQ(env->op_nodes_.size(), 2);
	ASSERT_EQ(env->in_nodes_.size(), 1);
	ASSERT_EQ(env->out_nodes_.size(), 1);

	ASSERT_EQ(env->in_nodes_[0], &env->tensor_nodes_[inn_id]);
	ASSERT_EQ(env->in_nodes_[0]->outputs_.size(), 1);
	ASSERT_EQ(env->in_nodes_[0]->outputs_[0], &env->op_nodes_[op1n_id]);

	ASSERT_EQ(env->op_nodes_[op1n_id].deps_.size(), 2);
	ASSERT_EQ(env->op_nodes_[op1n_id].deps_[0].ten_node_, &env->tensor_nodes_[inn_id]);
	ASSERT_EQ(env->op_nodes_[op1n_id].deps_[1].ten_node_, &env->tensor_nodes_[data1n_id]);
	ASSERT_EQ(env->op_nodes_[op1n_id].out_, &env->tensor_nodes_[flown_id]);

	ASSERT_EQ(env->tensor_nodes_[flown_id].outputs_.size(), 1);
	ASSERT_EQ(env->tensor_nodes_[flown_id].outputs_[0], &env->op_nodes_[op2n_id]);

	ASSERT_EQ(env->op_nodes_[op2n_id].deps_.size(), 2);
	ASSERT_EQ(env->op_nodes_[op2n_id].deps_[0].ten_node_, &env->tensor_nodes_[flown_id]);
	ASSERT_EQ(env->op_nodes_[op2n_id].deps_[1].ten_node_, &env->tensor_nodes_[data2n_id]);
	ASSERT_EQ(env->op_nodes_[op2n_id].out_, &env->tensor_nodes_[outn_id]);

	ASSERT_EQ(env->out_nodes_[0], &env->tensor_nodes_[outn_id]);
	ASSERT_EQ(env->out_nodes_[0]->outputs_.size(), 0);
	
}




TEST(CpuExecutor, Execute) {
	
	call_graph_builder cg_builder;
	auto inn_id = cg_builder.add_input_node(shape{768});
	auto data1n_id = cg_builder.add_data_node(shape{768, 128});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(noop{}, {inn_id, data1n_id}, shape{128});
	auto data2n_id = cg_builder.add_data_node(shape{128, 10});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(noop{}, {flown_id, data2n_id}, shape{10});
	cg_builder.make_output(outn_id);
	auto cg = cg_builder.build();

	hash_map<node_id, cpu_tensor> data_tensors;
	for (auto& [id, node]: cg.data_nodes_) {
		data_tensors[id] = cpu_tensor_factory::allocate(node.shape_);
	}

	cpu_exec_env_builder builder(cg);
	auto env = builder .alloc_flow_mem()
		.load_data_nodes(data_tensors)
		.build();

	cpu_tensor in_tensor = cpu_tensor_factory::allocate(shape{768});
	env->reset({in_tensor});
	ASSERT_EQ(env->unready_out_tens_, 1);
	ASSERT_EQ(env->state_, env_state::IN_PROGRESS);
	ASSERT_EQ(env->ready_q_.size(), 1);
	ASSERT_EQ(env->ready_q_.front(), &env->op_nodes_[op1n_id]);

	auto op = env->pop_ready_op();
	env->flow_node_ready(op->out_);
	ASSERT_EQ(env->unready_out_tens_, 1);
	ASSERT_EQ(env->state_, env_state::IN_PROGRESS);
	ASSERT_EQ(env->ready_q_.size(), 1);
	ASSERT_EQ(env->ready_q_.front(), &env->op_nodes_[op2n_id]);

	op = env->pop_ready_op();
	env->flow_node_ready(op->out_);
	ASSERT_EQ(env->unready_out_tens_, 0);
	ASSERT_EQ(env->state_, env_state::READY);
	ASSERT_EQ(env->ready_q_.size(), 0);

	auto output = env->output_tensors();
	ASSERT_EQ(output.size(), 1);
	ASSERT_EQ(output[0].meta_data().shape_, shape{10});
}

}

