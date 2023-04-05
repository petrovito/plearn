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
}

}

