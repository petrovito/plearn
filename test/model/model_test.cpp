#include "rep/ops.h"
#include <gtest/gtest.h>

#include <model/model.h>

namespace plearn::model {


TEST(Model, Execute) {
	Model m;
	auto input = m.add_input({3});
	auto variable = m.add_variable({3});

	auto difference = m.add_operation(operation{}, {3}, input, variable);
	auto output = m.add_operation(operation{}, {3}, difference);
	m.set_output(output);

	m.compile();

}

}

