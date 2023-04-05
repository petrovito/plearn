#include "backend/cpu/cpu_backend.h"
#include "backend/cpu/cpu_op_impl.h"
#include "backend/cpu/cpu_types.h"
#include "environ/env_types.h"
#include "environ/exec_env.h"
#include "rep/rep_types.h"
#include <gtest/gtest.h>

#include <backend/cpu/cpu_bw_grad.h>


namespace plearn::backend::cpu {

static const int dim1 = 256;
static const int dim2 = 128;
static const int SIZE = dim1*dim2;

static cpu_backend backend;
static exec_env env(&backend);

TEST(CpuBwGrad, VecMatmul) {
	tensor_p ten_a = env.create_tensor(shape_t{dim1});
	tensor_p ten_b = env.create_tensor(shape_t{dim1, dim2});
	tensor_p ten_c = env.create_tensor(shape_t{dim2});

	auto a = ((cpu_tensor*)ten_a->back())->get_content()->buf;
	auto b = ((cpu_tensor*)ten_b->back())->get_content()->buf;
	auto c = ((cpu_tensor*)ten_c->back())->get_content()->buf;
	std::fill_n(a, dim1, 1.f);
	std::fill_n(b, SIZE, 2.f);
	std::fill_n(c, dim2, 0.f);

	_cpu_vecmatmul(a, b, c, dim1, dim2);

	gradient in1_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient in2_grad = gradient{shape_t{dim1, dim2}, shape_t{1}, backend.create_tensor(shape_t{dim1, dim2, 1})};
	gradient out_grad = gradient{shape_t{dim2}, shape_t{1}, backend.create_tensor(shape_t{dim2, 1})};

	auto in1_grad_buf = ((cpu_tensor*)in1_grad.back_.get())->get_content()->buf;
	auto in2_grad_buf = ((cpu_tensor*)in2_grad.back_.get())->get_content()->buf;
	auto out_grad_buf = ((cpu_tensor*)out_grad.back_.get())->get_content()->buf;

	std::fill_n(in1_grad_buf, dim1, 0.f);
	std::fill_n(in2_grad_buf, SIZE, 0.f);
	std::fill_n(out_grad_buf, dim2, 3.f);

	cpu_bw_vecmatmul bw;
	vector<tensor_p> inputs = {ten_a, ten_b};
	bw.reset(inputs, ten_c);

	bw.update_grad(0, out_grad, in1_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in1_grad_buf[i], 2*3*dim2);
	}

	bw.update_grad(1, out_grad, in2_grad);
	for (int i = 0; i < dim1; i++) {
		for (int j = 0; j < dim2; j++) {
			EXPECT_EQ(in2_grad_buf[i*dim2+j], 1*3);
		}
	}
}

TEST(CpuBwGrad, Square) {
	tensor_p ten_a = env.create_tensor(shape_t{dim1});
	tensor_p ten_b = env.create_tensor(shape_t{dim1});

	auto a = ((cpu_tensor*)ten_a->back())->get_content()->buf;
	auto b = ((cpu_tensor*)ten_b->back())->get_content()->buf;
	std::fill_n(a, dim1, 1.f);
	std::fill_n(b, dim1, 0.f);

	_cpu_square(a, b, dim1);

	gradient in_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient out_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};

	auto in_grad_buf = ((cpu_tensor*)in_grad.back_.get())->get_content()->buf;
	auto out_grad_buf = ((cpu_tensor*)out_grad.back_.get())->get_content()->buf;

	std::fill_n(in_grad_buf, dim1, 0.f);
	std::fill_n(out_grad_buf, dim1, 3.f);

	cpu_bw_square bw;
	vector<tensor_p> inputs = {ten_a};
	bw.reset(inputs, ten_b);

	bw.update_grad(0, out_grad, in_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in_grad_buf[i], 2*3);
	}
}

TEST(CpuBwGrad, Add) {
	tensor_p ten_a = env.create_tensor(shape_t{dim1});
	tensor_p ten_b = env.create_tensor(shape_t{dim1});
	tensor_p ten_c = env.create_tensor(shape_t{dim1});

	auto a = ((cpu_tensor*)ten_a->back())->get_content()->buf;
	auto b = ((cpu_tensor*)ten_b->back())->get_content()->buf;
	auto c = ((cpu_tensor*)ten_c->back())->get_content()->buf;
	std::fill_n(a, dim1, 1.f);
	std::fill_n(b, dim1, 2.f);
	std::fill_n(c, dim1, 0.f);

	_cpu_add(a, b, c, dim1);

	gradient in1_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient in2_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient out_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};

	auto in1_grad_buf = ((cpu_tensor*)in1_grad.back_.get())->get_content()->buf;
	auto in2_grad_buf = ((cpu_tensor*)in2_grad.back_.get())->get_content()->buf;
	auto out_grad_buf = ((cpu_tensor*)out_grad.back_.get())->get_content()->buf;

	std::fill_n(in1_grad_buf, dim1, 0.f);
	std::fill_n(in2_grad_buf, dim1, 0.f);
	std::fill_n(out_grad_buf, dim1, 3.f);

	cpu_bw_add bw;
	vector<tensor_p> inputs = {ten_a, ten_b};
	bw.reset(inputs, ten_c);

	bw.update_grad(0, out_grad, in1_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in1_grad_buf[i], 3);
	}

	bw.update_grad(1, out_grad, in2_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in2_grad_buf[i], 3);
	}
}

TEST(CpuBwGrad, Sub) {
	tensor_p ten_a = env.create_tensor(shape_t{dim1});
	tensor_p ten_b = env.create_tensor(shape_t{dim1});
	tensor_p ten_c = env.create_tensor(shape_t{dim1});

	auto a = ((cpu_tensor*)ten_a->back())->get_content()->buf;
	auto b = ((cpu_tensor*)ten_b->back())->get_content()->buf;
	auto c = ((cpu_tensor*)ten_c->back())->get_content()->buf;
	std::fill_n(a, dim1, 1.f);
	std::fill_n(b, dim1, 2.f);
	std::fill_n(c, dim1, 0.f);

	_cpu_sub(a, b, c, dim1);

	gradient in1_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient in2_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient out_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};

	auto in1_grad_buf = ((cpu_tensor*)in1_grad.back_.get())->get_content()->buf;
	auto in2_grad_buf = ((cpu_tensor*)in2_grad.back_.get())->get_content()->buf;
	auto out_grad_buf = ((cpu_tensor*)out_grad.back_.get())->get_content()->buf;

	std::fill_n(in1_grad_buf, dim1, 0.f);
	std::fill_n(in2_grad_buf, dim1, 0.f);
	std::fill_n(out_grad_buf, dim1, 3.f);

	cpu_bw_sub bw;
	vector<tensor_p> inputs = {ten_a, ten_b};
	bw.reset(inputs, ten_c);

	bw.update_grad(0, out_grad, in1_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in1_grad_buf[i], 3);
	}

	bw.update_grad(1, out_grad, in2_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in2_grad_buf[i], -3);
	}
}

TEST(CpuBwGrad, Mult) {
	tensor_p ten_a = env.create_tensor(shape_t{dim1});
	tensor_p ten_b = env.create_tensor(shape_t{dim1});
	tensor_p ten_c = env.create_tensor(shape_t{dim1});

	auto a = ((cpu_tensor*)ten_a->back())->get_content()->buf;
	auto b = ((cpu_tensor*)ten_b->back())->get_content()->buf;
	auto c = ((cpu_tensor*)ten_c->back())->get_content()->buf;
	std::fill_n(a, dim1, 1.f);
	std::fill_n(b, dim1, 2.f);
	std::fill_n(c, dim1, 0.f);

	_cpu_mult(a, b, c, dim1);

	gradient in1_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient in2_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};
	gradient out_grad = gradient{shape_t{dim1}, shape_t{1}, backend.create_tensor(shape_t{dim1, 1})};

	auto in1_grad_buf = ((cpu_tensor*)in1_grad.back_.get())->get_content()->buf;
	auto in2_grad_buf = ((cpu_tensor*)in2_grad.back_.get())->get_content()->buf;
	auto out_grad_buf = ((cpu_tensor*)out_grad.back_.get())->get_content()->buf;

	std::fill_n(in1_grad_buf, dim1, 0.f);
	std::fill_n(in2_grad_buf, dim1, 0.f);
	std::fill_n(out_grad_buf, dim1, 3.f);

	cpu_bw_mult bw;
	vector<tensor_p> inputs = {ten_a, ten_b};
	bw.reset(inputs, ten_c);

	bw.update_grad(0, out_grad, in1_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in1_grad_buf[i], 3*2);
	}

	bw.update_grad(1, out_grad, in2_grad);
	for (int i = 0; i < dim1; i++) {
		EXPECT_EQ(in2_grad_buf[i], 3*1);
	}
}
}

