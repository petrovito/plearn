#pragma once
#include <call_graph.h>

using namespace plearn;

inline call_graph call_graph_example() {

	call_graph_builder builder;

	auto inn_id = builder.add_input_node(shape{768});
	auto data1n_id = builder.add_data_node(shape{768, 128});
	auto [op1n_id, flown_id] = 
		builder.add_op_node(noop{}, {inn_id, data1n_id}, shape{128});
	auto data2n_id = builder.add_data_node(shape{128, 10});
	auto [op2n_id, outn_id] = 
		builder.add_op_node(noop{}, {flown_id, data2n_id}, shape{10});
	builder.make_output(outn_id);

	return builder.build();
}

