#include <algorithm>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <new>

#include <backend/cpu/cpu_ops.h>

using namespace plearn::backend::cpu;

const int dim = 256;
const int SIZE = dim*dim;

TEST(CpuOpTest, Matmul) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, SIZE, 1);
	std::fill_n(c, SIZE, 0);

	_cpu_matmul(a, b, c, dim, dim, dim);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(dim, c[i*dim +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}


TEST(CpuOpTest, MatVecMul) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim];
	auto c = new (std::align_val_t(32)) float[dim];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, dim, 1);
	std::fill_n(c, dim, 0);

	_cpu_matvecmul(a, b, c, dim, dim);
	for (int i = 0; i < dim; i++)
		ASSERT_DOUBLE_EQ(dim, c[i]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, Add) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, SIZE, 2);
	std::fill_n(c, SIZE, 0);

	_cpu_add(a, b, c, SIZE);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(3, c[i*dim +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, Sub) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, SIZE, 2);
	std::fill_n(c, SIZE, 0);

	_cpu_sub(a, b, c, SIZE);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(-1, c[i*dim +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, Square) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 2);
	std::fill_n(b, SIZE, 0);

	_cpu_square(a, b, SIZE);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(4, b[i*dim +j]);

	delete[] a;
	delete[] b;
}

TEST(CpuOpTest, Mult) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 2);
	std::fill_n(b, SIZE, 3);
	std::fill_n(c, SIZE, 0);

	_cpu_mult(a, b, c, SIZE);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(6, c[i*dim +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, ReduceSum) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim];
	std::fill_n(a, SIZE, 2.f);
	std::fill_n(b, dim, 0.f);

	_cpu_reduce_sum(a, b, 1, {dim, dim});
	for (int i = 0; i < dim; i++)
		ASSERT_DOUBLE_EQ(2*dim, b[i]);

	delete[] a;
	delete[] b;
}

TEST(CpuOpTest, ReduceMean) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim];
	std::fill_n(a, SIZE, 2.f);
	std::fill_n(b, dim, 0.f);

	_cpu_reduce_mean(a, b, 1, {dim, dim});
	for (int i = 0; i < dim; i++)
		ASSERT_DOUBLE_EQ(2, b[i]);

	delete[] a;
	delete[] b;
}

