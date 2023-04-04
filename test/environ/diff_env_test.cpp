#include <gtest/gtest.h>

#include "backend_mocks.h"

#include "rep/call_graph.h"
#include "rep/diff_info.h"
#include <environ/fp_diff_env.h>
#include <environ/bw_diff_env.h>

namespace plearn::env {



TEST(BwDiffEnv, Builder) {
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


	auto diff_graph_builder = diff_info_builder(cg);
	auto diff_graph = diff_graph_builder
		.all_data_nodes().find_dependencies().build();

	auto mock_backend = std::make_unique<MockBackend>();

	auto bw_de_builder = bw_diff_env_builder(cg, diff_graph.get(), mock_backend.get());
	auto bw_diff_env = bw_de_builder.allocate_grad_tensors().build();

	auto grad_sys = bw_diff_env->get_grad_system();

	ASSERT_EQ(grad_sys->size(), 5); //outn + flown + 2* datan + 1*inn

	ASSERT_EQ((*grad_sys)[outn_id].size(), 1);
	ASSERT_EQ((*grad_sys)[flown_id].size(), 1);
	ASSERT_EQ((*grad_sys)[data1n_id].size(), 1);
	ASSERT_EQ((*grad_sys)[data2n_id].size(), 1);
	ASSERT_EQ((*grad_sys)[inn_id].size(), 0);


#define MOCK_TENSOR_BACK(node_id) ((MockTensorBack*)(*grad_sys)[node_id][outn_id].grad_.back_.get())
	ASSERT_EQ(MOCK_TENSOR_BACK(outn_id)->shape_, shape_t(1,1));
	ASSERT_EQ(MOCK_TENSOR_BACK(flown_id)->shape_, shape_t(3,1));
	ASSERT_EQ(MOCK_TENSOR_BACK(data1n_id)->shape_, shape_t(2,3,1));
	ASSERT_EQ(MOCK_TENSOR_BACK(data2n_id)->shape_, shape_t(3,1,1));
#undef MOCK_TENSOR_BACK

	//TODO test identity of outn grad
	//TODO test op diff envs?
}

}

