#include "backend/cpu/cpu_types.h"
#include <gtest/gtest.h>

#include <environ/env_types.h>
#include <environ/env_section.h>
#include <environ/exec_env.h>
#include <backend/cpu/cpu_env.h>

namespace plearn::backend::cpu {

TEST(CpuBackendIntegration, Execute) {

	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{2});
	auto data1n_id = cg_builder.add_data_node(shape_t{2, 3});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(vecmatmul{}, {inn_id, data1n_id}, shape_t{128});
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

	env_section section{env.get(), backend.get(), cg, std::move(data_tensors)};

	exec_params params;
	params.inputs_[inn_id] = input_ten;
	params.outputs_[outn_id] = env->create_tensor(cg.flow_nodes_.at(outn_id).shape_);

	auto result = section.execute(params);

	auto buf = ((cpu_tensor*) params.outputs_.at(outn_id)->back())->get_content()->buf;
	//internal node = [9,12,15]
	EXPECT_FLOAT_EQ(buf[0], 78);
}

}

