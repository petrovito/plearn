#include <algorithm>
#include <gtest/gtest.h>
#include <cpu_ops.h>
#include <immintrin.h>
#include <new>

using namespace plearn;

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
}

TEST(CpuOpTest, MatmulAvx) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, SIZE, 1);
	std::fill_n(c, SIZE, 0);

	_cpu_matmul_avx(a, b, c, dim, dim, dim);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			ASSERT_DOUBLE_EQ(dim, c[i*dim +j]);
}

