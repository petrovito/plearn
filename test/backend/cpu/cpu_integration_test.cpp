#include <gtest/gtest.h>

#include "backend/cpu/cpu_types.h"
#include <environ/env_types.h>
#include <environ/env_section.h>
#include <environ/exec_env.h>
#include <backend/cpu/cpu_backend.h>

namespace plearn::backend::cpu {

TEST(CpuBackendIntegration, Execute) {

	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{2});
	auto data1n_id = cg_builder.add_data_node(shape_t{2, 3});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(vecmatmul{}, {inn_id, data1n_id}, shape_t{3});
	auto data2n_id = cg_builder.add_data_node(shape_t{3, 1});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(vecmatmul{}, {flown_id, data2n_id}, shape_t{1});
	cg_builder.make_output(outn_id);

	auto cg = cg_builder.build();


	unique_ptr<cpu_backend> backend = std::make_unique<cpu_backend>();
	unique_ptr<exec_env> env = std::make_unique<exec_env>(backend.get());

	hash_map<node_id, tensor_p> data_tensors;
	data_tensors[data1n_id] = env->create_tensor(cg.data_nodes_.at(data1n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data1n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
		buf[3] = 4;
		buf[4] = 5;
		buf[5] = 6;
	}
	data_tensors[data2n_id] = env->create_tensor(cg.data_nodes_.at(data2n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data2n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
	}

	auto input_ten = env->create_tensor(cg.flow_nodes_.at(inn_id).shape_);
	{
		auto buf = ((cpu_tensor*) input_ten->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
	}

	env_section_builder section_builder{env.get(), backend.get(), cg};
	auto section = section_builder
		.allocate_internal_tensors()
		.set_data_tensors(std::move(data_tensors))
		.build();

	

	{
	exec_params params;
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	//internal node = [9,12,15]
	EXPECT_FLOAT_EQ(buf[0], 78);
	}


	//run again
	{
	exec_params params;
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	//internal node = [9,12,15]
	EXPECT_FLOAT_EQ(buf[0], 78);
	}
}


TEST(CpuBackendIntegration, DiffFw) {
	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{2});
	auto data1n_id = cg_builder.add_data_node(shape_t{2, 3});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(vecmatmul{}, {inn_id, data1n_id}, shape_t{3});
	auto data2n_id = cg_builder.add_data_node(shape_t{3, 1});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(vecmatmul{}, {flown_id, data2n_id}, shape_t{1});
	cg_builder.make_output(outn_id);

	auto cg = cg_builder.build();


	unique_ptr<cpu_backend> backend = std::make_unique<cpu_backend>();
	unique_ptr<exec_env> env = std::make_unique<exec_env>(backend.get());

	hash_map<node_id, tensor_p> data_tensors;
	data_tensors[data1n_id] = env->create_tensor(cg.data_nodes_.at(data1n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data1n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
		buf[3] = 4;
		buf[4] = 5;
		buf[5] = 6;
	}
	data_tensors[data2n_id] = env->create_tensor(cg.data_nodes_.at(data2n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data2n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
	}

	auto input_ten = env->create_tensor(cg.flow_nodes_.at(inn_id).shape_);
	{
		auto buf = ((cpu_tensor*) input_ten->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
	}

	env_section_builder section_builder{env.get(), backend.get(), cg};
	auto section = section_builder
		.allocate_internal_tensors()
		.set_data_tensors(std::move(data_tensors))
		.create_diff_info()
		.create_fp_diff()
		.build();

	{
	exec_params params{.calc_diffs = true};
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	//internal node = [9,12,15]
	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	EXPECT_FLOAT_EQ(buf[0], 78);

	auto& grads = result.grad_system_->at(outn_id);
	auto data1_out_grad = grads.at(data1n_id);
	auto data1_out_grad_buf = ((cpu_tensor*) data1_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data1_out_grad_buf[0], 1);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[1], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[2], 3);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[3], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[4], 4);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[5], 6);

	auto data2_out_grad = grads.at(data2n_id);
	auto data2_out_grad_buf = ((cpu_tensor*) data2_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data2_out_grad_buf[0], 9);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[1], 12);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[2], 15);
	}
	
	//run again
	{
	exec_params params{.calc_diffs = true};
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	//internal node = [9,12,15]
	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	EXPECT_FLOAT_EQ(buf[0], 78);

	auto& grads = result.grad_system_->at(outn_id);
	auto data1_out_grad = grads.at(data1n_id);
	auto data1_out_grad_buf = ((cpu_tensor*) data1_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data1_out_grad_buf[0], 1);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[1], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[2], 3);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[3], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[4], 4);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[5], 6);

	auto data2_out_grad = grads.at(data2n_id);
	auto data2_out_grad_buf = ((cpu_tensor*) data2_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data2_out_grad_buf[0], 9);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[1], 12);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[2], 15);
	}
}


TEST(CpuBackendIntegration, DiffFw2) {
	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{1,2});
	auto data1n_id = cg_builder.add_data_node(shape_t{2, 3});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape_t{1,3});
	auto data2n_id = cg_builder.add_data_node(shape_t{3, 1});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape_t{1,1});
	cg_builder.make_output(outn_id);

	auto cg = cg_builder.build();


	unique_ptr<cpu_backend> backend = std::make_unique<cpu_backend>();
	unique_ptr<exec_env> env = std::make_unique<exec_env>(backend.get());

	hash_map<node_id, tensor_p> data_tensors;
	data_tensors[data1n_id] = env->create_tensor(cg.data_nodes_.at(data1n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data1n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
		buf[3] = 4;
		buf[4] = 5;
		buf[5] = 6;
	}
	data_tensors[data2n_id] = env->create_tensor(cg.data_nodes_.at(data2n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data2n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
	}

	auto input_ten = env->create_tensor(cg.flow_nodes_.at(inn_id).shape_);
	{
		auto buf = ((cpu_tensor*) input_ten->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
	}

	env_section_builder section_builder{env.get(), backend.get(), cg};
	auto section = section_builder
		.allocate_internal_tensors()
		.set_data_tensors(std::move(data_tensors))
		.create_diff_info()
		.create_fp_diff()
		.build();

	exec_params params{.calc_diffs = true};
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	//internal node = [9,12,15]
	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	EXPECT_FLOAT_EQ(buf[0], 78);

	auto& grads = result.grad_system_->at(outn_id);
	auto data1_out_grad = grads.at(data1n_id);
	auto data1_out_grad_buf = ((cpu_tensor*) data1_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data1_out_grad_buf[0], 1);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[1], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[2], 3);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[3], 2);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[4], 4);
	EXPECT_FLOAT_EQ(data1_out_grad_buf[5], 6);

	auto data2_out_grad = grads.at(data2n_id);
	auto data2_out_grad_buf = ((cpu_tensor*) data2_out_grad.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data2_out_grad_buf[0], 9);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[1], 12);
	EXPECT_FLOAT_EQ(data2_out_grad_buf[2], 15);
}


TEST(CpuBackendIntegration, DiffBw) {
	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{2});
	auto data1n_id = cg_builder.add_data_node(shape_t{2, 3});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(vecmatmul{}, {inn_id, data1n_id}, shape_t{3});
	auto data2n_id = cg_builder.add_data_node(shape_t{3, 1});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(vecmatmul{}, {flown_id, data2n_id}, shape_t{1});
	cg_builder.make_output(outn_id);

	auto cg = cg_builder.build();


	unique_ptr<cpu_backend> backend = std::make_unique<cpu_backend>();
	unique_ptr<exec_env> env = std::make_unique<exec_env>(backend.get());

	hash_map<node_id, tensor_p> data_tensors;
	data_tensors[data1n_id] = env->create_tensor(cg.data_nodes_.at(data1n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data1n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
		buf[3] = 4;
		buf[4] = 5;
		buf[5] = 6;
	}
	data_tensors[data2n_id] = env->create_tensor(cg.data_nodes_.at(data2n_id).shape_);
	{
		auto buf = ((cpu_tensor*) data_tensors[data2n_id]->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
		buf[2] = 3;
	}

	auto input_ten = env->create_tensor(cg.flow_nodes_.at(inn_id).shape_);
	{
		auto buf = ((cpu_tensor*) input_ten->back())->get_content()->buf;
		buf[0] = 1;
		buf[1] = 2;
	}

	env_section_builder section_builder{env.get(), backend.get(), cg};
	auto section = section_builder
		.allocate_internal_tensors()
		.set_data_tensors(std::move(data_tensors))
		.create_diff_info()
		.create_bw_diff()
		.build();

	for (int i = 0; i < 2; ++i)
	{
	exec_params params{.calc_diffs = true};
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	//internal node = [9,12,15]
	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	EXPECT_FLOAT_EQ(buf[0], 78);

	auto& data1_grads = result.grad_system_->at(data1n_id).at(outn_id);
	auto data1_grad_buf = ((cpu_tensor*) data1_grads.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data1_grad_buf[0], 1);
	EXPECT_FLOAT_EQ(data1_grad_buf[1], 2);
	EXPECT_FLOAT_EQ(data1_grad_buf[2], 3);
	EXPECT_FLOAT_EQ(data1_grad_buf[3], 2);
	EXPECT_FLOAT_EQ(data1_grad_buf[4], 4);
	EXPECT_FLOAT_EQ(data1_grad_buf[5], 6);

	auto& data2_grads = result.grad_system_->at(data2n_id).at(outn_id);
	auto data2_grad_buf = ((cpu_tensor*) data2_grads.grad_.back_.get())->get_content()->buf;
	EXPECT_FLOAT_EQ(data2_grad_buf[0], 9);
	EXPECT_FLOAT_EQ(data2_grad_buf[1], 12);
	EXPECT_FLOAT_EQ(data2_grad_buf[2], 15);
	}
	
	//run again
}


}

