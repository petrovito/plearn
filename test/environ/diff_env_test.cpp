#include <algorithm>
#include <gmock/gmock-spec-builders.h>
#include <gtest/gtest.h>

#include "backend_mocks.h"

#include "environ/env_types.h"
#include "rep/call_graph.h"
#include "rep/diff_info.h"
#include "rep/rep_types.h"
#include <environ/fp_diff_env.h>
#include <environ/bw_diff_page.h>
#include <memory>

namespace plearn::env {

using testing::_;

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

	auto bw_de_builder = bw_diff_page_builder(cg, diff_graph.get(), mock_backend.get());
	auto bw_diff_page = bw_de_builder.build();

	auto grad_sys = bw_diff_page->get_grad_system();

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

	//test outnode is identity
	ASSERT_TRUE((*grad_sys)[outn_id][outn_id].identity_);
	ASSERT_EQ(((MockTensorBack*)(*grad_sys)[outn_id][outn_id].grad_.back_.get())->init_mode_, 
			tensor_init::identity);
}

TEST(BwDiffEnv, OpDiffEnvExecute) {
	auto exec_env = MockExecEnv();
	unique_ptr<MockBwOpDiffBackend> mock_diff_backend = std::make_unique<MockBwOpDiffBackend>();
	
	node_id outn_id = 999;

	grad_map out_grad_map;
	shape_t out_shape{2, 1};
	out_grad_map[outn_id] = {3, outn_id, {out_shape, {1}}};

	vector<grad_map*> in_grad_maps;
	auto in_grad_map1 = std::make_unique<grad_map>();
	shape_t in1_shape{2, 3};
	(*in_grad_map1)[outn_id] = {1, outn_id, {in1_shape, {1}}};
	in_grad_maps.push_back(in_grad_map1.get());

	auto in_grad_map2 = std::make_unique<grad_map>();
	shape_t in2_shape{3, 1};
	(*in_grad_map2)[outn_id] = {2, outn_id, {in2_shape, {1}}};
	in_grad_maps.push_back(in_grad_map2.get());

	EXPECT_CALL(*mock_diff_backend, reset(_,_));
	EXPECT_CALL(*mock_diff_backend, update_grad(_,_,_)).Times(2);

	bw_op_diff_env bw_op_de{std::move(mock_diff_backend), &out_grad_map, std::move(in_grad_maps)};

	auto in1 = exec_env.create_tensor(shape_t{2, 3});
	auto in2 = exec_env.create_tensor(shape_t{3, 1});
	auto out = exec_env.create_tensor(shape_t{2, 1});

	bw_op_de.execute({in1, in2}, out);
}

}

