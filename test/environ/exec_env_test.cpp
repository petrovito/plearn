#include "rep/rep_types.h"
#include <gmock/gmock-function-mocker.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <environ/env_types.h>
#include <environ/env_section.h>
#include <environ/exec_env.h>
#include <memory>

namespace plearn::env {

	using ::testing::_;


class MockBackend : public backend_t {
	public:
		MOCK_METHOD(void, exec_op,
				(const operation& op, const vector<tensor_p>& inputs, tensor_p& output)
				, (override));

		unique_ptr<tensor_back_t> create_tensor(const shape_t&) override {
			return std::unique_ptr<tensor_back_t>(nullptr);
		}

		void calc_forward_grad(const operation& ,
				const vector<tensor_p>& , const tensor_p& ,
				const vector<read_ptr<grad_map>>& , grad_map& ) override {};
};

class MockExecEnv : public exec_env {
	public:
		MockExecEnv(backend_t* backend) : exec_env(backend) {}
		tensor_p create_tensor(const shape_t& s) override {
			auto back_tensor = std::unique_ptr<tensor_back_t>(nullptr);
			return tens_fac_.create(s, std::move(back_tensor));
		}
};


TEST(EnvSection, Execute) {

	call_graph_builder cg_builder;

	auto inn_id = cg_builder.add_input_node(shape_t{768});
	auto data1n_id = cg_builder.add_data_node(shape_t{768, 128});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(vecmatmul{}, {inn_id, data1n_id}, shape_t{128});
	auto data2n_id = cg_builder.add_data_node(shape_t{128, 10});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(vecmatmul{}, {flown_id, data2n_id}, shape_t{10});
	cg_builder.make_output(outn_id);

	auto cg = cg_builder.build();


	unique_ptr<MockBackend> mock_backend = std::make_unique<MockBackend>();
	unique_ptr<MockExecEnv> mock_env = std::make_unique<MockExecEnv>(mock_backend.get());

	hash_map<node_id, tensor_p> data_tensors;
	for (auto& [id, node]: cg.data_nodes_) {
		data_tensors[id] = mock_env->create_tensor(node.shape_);
	}
	
	env_section section{mock_env.get(), mock_backend.get(), cg, std::move(data_tensors)};

	exec_params params;
	params.inputs_[inn_id] = mock_env->create_tensor(shape_t{768});
	params.outputs_[outn_id] = mock_env->create_tensor(shape_t{10});

	EXPECT_CALL(*mock_backend, exec_op(_, _, _)).Times(2);
	auto result = section.execute(params);
	ASSERT_TRUE(result.success);

	ASSERT_EQ(params.outputs_[outn_id]->shape(), shape_t{10});
}

}

