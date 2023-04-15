#include <algorithm>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <immintrin.h>
#include <new>

#include <backend/cpu/cpu_ops.h>

using namespace plearn::backend::cpu;

static const int dim1 = 256;
static const int dim2 = 128;
static const int dim3 = 64;

static const int SIZE = dim1*dim1;
static const int SIZE1 = dim1*dim2;
static const int SIZE2 = dim2*dim3;
static const int SIZE3 = dim1*dim3;

TEST(CpuOpTest, Matmul) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[SIZE];
	auto c = new (std::align_val_t(32)) float[SIZE];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, SIZE, 1);
	std::fill_n(c, SIZE, 0);

	_cpu_matmul(a, b, c, dim1, dim1, dim1);
	for (int i = 0; i < dim1; i++)
		for (int j = 0; j < dim1; j++)
			ASSERT_DOUBLE_EQ(dim1, c[i*dim1 +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, MatMulTranspose) {
	int dim1 = 3;
	int dim2 = 2;
	int dim3 = 4;
	int SIZE1 = dim1*dim2;
	int SIZE2 = dim3*dim2;
	int SIZE3 = dim1*dim3;

	auto a = new (std::align_val_t(32)) float[SIZE1] {
		1, 2,
		3, 4,
		5, 6
	};
	auto b = new (std::align_val_t(32)) float[SIZE2] {
		1, 2,
		3, 4,
		5, 6,
		7, 8
	};
	auto c = new (std::align_val_t(32)) float[SIZE3];

	_cpu_matmul(a, b, c, dim1, dim2, dim3, false, false, true);
	ASSERT_THAT(vector<float>(c, c+SIZE3), testing::ElementsAre(
				5, 11, 17, 23,
				11, 25, 39, 53,
				17, 39, 61, 83
				));

	delete[] a;
	a = new (std::align_val_t(32)) float[SIZE1] {
		1, 3, 5,
		2, 4, 6
	};

	_cpu_matmul(a, b, c, dim1, dim2, dim3, false, true, true);
	ASSERT_THAT(vector<float>(c, c+SIZE3), testing::ElementsAre(
				5, 11, 17, 23,
				11, 25, 39, 53,
				17, 39, 61, 83
				));

	delete[] b;
	b = new (std::align_val_t(32)) float[SIZE2] {
		1, 3, 5, 7,
		2, 4, 6, 8
	};

	_cpu_matmul(a, b, c, dim1, dim2, dim3, false, true, false);
	ASSERT_THAT(vector<float>(c, c+SIZE3), testing::ElementsAre(
				5, 11, 17, 23,
				11, 25, 39, 53,
				17, 39, 61, 83
				));

	delete[] a;
	delete[] b;
	delete[] c;
}


TEST(CpuOpTest, MatVecMul) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim1];
	auto c = new (std::align_val_t(32)) float[dim1];
	std::fill_n(a, SIZE, 1);
	std::fill_n(b, dim1, 1);
	std::fill_n(c, dim1, 0);

	_cpu_matvecmul(a, b, c, dim1, dim1);
	for (int i = 0; i < dim1; i++)
		ASSERT_DOUBLE_EQ(dim1, c[i]);

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
	for (int i = 0; i < dim1; i++)
		for (int j = 0; j < dim1; j++)
			ASSERT_DOUBLE_EQ(3, c[i*dim1 +j]);

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
	for (int i = 0; i < dim1; i++)
		for (int j = 0; j < dim1; j++)
			ASSERT_DOUBLE_EQ(-1, c[i*dim1 +j]);

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
	for (int i = 0; i < dim1; i++)
		for (int j = 0; j < dim1; j++)
			ASSERT_DOUBLE_EQ(4, b[i*dim1 +j]);

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
	for (int i = 0; i < dim1; i++)
		for (int j = 0; j < dim1; j++)
			ASSERT_DOUBLE_EQ(6, c[i*dim1 +j]);

	delete[] a;
	delete[] b;
	delete[] c;
}

TEST(CpuOpTest, ReduceSum) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim1];
	std::fill_n(a, SIZE, 2.f);
	std::fill_n(b, dim1, 0.f);

	_cpu_reduce_sum(a, b, 1, {dim1, dim1});
	for (int i = 0; i < dim1; i++)
		ASSERT_DOUBLE_EQ(2*dim1, b[i]);

	delete[] a;
	delete[] b;
}

TEST(CpuOpTest, ReduceMean) {
	auto a = new (std::align_val_t(32)) float[SIZE];
	auto b = new (std::align_val_t(32)) float[dim1];
	std::fill_n(a, SIZE, 2.f);
	std::fill_n(b, dim1, 0.f);

	_cpu_reduce_mean(a, b, 1, {dim1, dim1});
	for (int i = 0; i < dim1; i++)
		ASSERT_DOUBLE_EQ(2, b[i]);

	delete[] a;
	delete[] b;
}

