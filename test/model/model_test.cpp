#include "rep/ops.h"
#include "rep/rep_types.h"
#include <gtest/gtest.h>

#include <model/model.h>

namespace plearn::model {


TEST(Model, Execute) {
	Model m;
	auto input = m.add_input({3});
	auto variable = m.add_variable({3});

	auto difference = input - variable;
	auto sq = difference.square();
	auto reduce = sq.reduce_sum(0);

	m.set_output(reduce);
	m.compile();

	variable.set_tensor(Tensors::create({3}, new float[]{1, 2, 3}));

	auto input_ten = Tensors::create({3}, new float[]{2,4,6});
	auto output = m.execute({input_ten}, true);

	auto out_data = output.tensors[0]->data();
	ASSERT_EQ(out_data[0], 1 + 4 + 9);

	auto& var_out_grad = output.grad_of(variable, reduce);
	ASSERT_EQ(var_out_grad.in_shape, shape_t{3});
	ASSERT_EQ(var_out_grad.out_shape, shape_t{1});
	auto out_diffs = var_out_grad.data();
	ASSERT_EQ(out_diffs[0], -2);
	ASSERT_EQ(out_diffs[1], -4);
	ASSERT_EQ(out_diffs[2], -6);

}

}

