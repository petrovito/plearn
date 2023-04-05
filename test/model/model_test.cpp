#include "rep/ops.h"
#include <gtest/gtest.h>

#include <model/model.h>

namespace plearn::model {


TEST(Model, Execute) {
	Model m;
	auto input = m.add_input({3});
	auto variable = m.add_variable({3});

	auto difference = input - variable;
	auto sq = difference.square();

	m.set_output(sq);
	m.compile();

	variable.set_tensor(Tensors::create({3}, new float[]{1, 2, 3}));

	auto input_ten = Tensors::create({3}, new float[]{2,4,6});
	auto output_tens = m.execute({input_ten});

	auto out_data = output_tens[0]->data();
	ASSERT_EQ(out_data[0], 1);
	ASSERT_EQ(out_data[1], 4);
	ASSERT_EQ(out_data[2], 9);
}

}

