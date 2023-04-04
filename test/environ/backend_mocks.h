#pragma once

#include <gmock/gmock.h>

#include <environ/env_types.h>
#include "environ/exec_env.h"
#include "rep/rep_types.h"

namespace plearn::env {

class MockTensorBack : public tensor_back_t {
	public:
		MockTensorBack(const shape_t& s) : shape_(s) {}
		void zero() override {};
		shape_t shape_;
};

class MockBackend : public backend_t {
	public:
		MOCK_METHOD(void, exec_op,
				(const operation& op, const vector<tensor_p>& inputs, tensor_p& output)
				, (override));

		unique_ptr<tensor_back_t> create_tensor(const shape_t& s) override {
			return std::make_unique<MockTensorBack>(s);
		}

		unique_ptr<fw_op_diff_backend_t> create_op_fw_diff_backend(
				const operation&  ) override { return nullptr; }

		unique_ptr<bw_op_diff_backend_t> create_op_bw_diff_backend(
				const operation&  ) override { return nullptr; }

};



class MockExecEnv : public exec_env {
	public:
		MockExecEnv() : exec_env(nullptr) {};
		MockExecEnv(backend_t* backend) : exec_env(backend) {}
		tensor_p create_tensor(const shape_t& s) override {
			auto back_tensor = std::make_unique<MockTensorBack>(s);
			return tens_fac_.create(s, std::move(back_tensor));
		}

};


class MockBwOpDiffBackend : public bw_op_diff_backend_t {
	public:
		MOCK_METHOD(void, reset,
				(const vector<tensor_p>& inputs, const tensor_p& output)
				, (override));
		MOCK_METHOD(void, update_grad,
				(unsigned input_idx, const gradient& out_grad, gradient& var_out_grad)
				, (override));
};


}
