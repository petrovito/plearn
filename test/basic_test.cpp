#include <iostream>
#include <operation.h>
#include <gtest/gtest.h>

using namespace plearn;

TEST(OpTest, Basic) {
	Tensor<S<1,2,3>> t1;
	std::cout << t1.shape.size() << "HEYYYYYYYYYYYYYY" << std::endl;
	Tensor<S<1,2,3>> t2;

	auto add_op = Add(t1, t2);
	auto t3 = add_op.apply();

	Tensor<S<233,9,2>> t4;
	Tensor<S<233,2,122>> t5;
	MatMul mul_op(t4, t5);
	auto x = mul_op.apply();

	/* auto test1 = firstElem<1,2,3>(); */
	/* auto test2 = allButFirst<1, 2, 3>(); */
	/* auto test3 = lastElem<1,2,3>(); */
	IntSeq<1,2,3> list;
	auto test_list = allButLast(list);
	auto arr = test_list.arr;

}

