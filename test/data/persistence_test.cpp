#include <gtest/gtest.h>
#include <string>

#include <rep/call_graph.h>
#include <data/persistence.h>

namespace plearn::data {

TEST(Persistence, SaveLoad) {
	std::string path = "/tmp/test.pb";
	
	call_graph_builder builder;

	auto inn_id = builder.add_input_node(shape_t{768});
	auto data1n_id = builder.add_data_node(shape_t{768, 128});
	auto [op1n_id, flown_id] = 
		builder.add_op_node(noop{}, {inn_id, data1n_id}, shape_t{128});
	auto data2n_id = builder.add_data_node(shape_t{128, 10});
	auto [op2n_id, outn_id] = 
		builder.add_op_node(noop{}, {flown_id, data2n_id}, shape_t{10});
	builder.make_output(outn_id);
	auto cg = builder.build();

	Persistence::save(cg, path);
	call_graph cg2 = Persistence::load(path);

	EXPECT_EQ(cg.flow_nodes_, cg2.flow_nodes_);
	EXPECT_EQ(cg.data_nodes_, cg2.data_nodes_);
	EXPECT_EQ(cg.op_nodes_, cg2.op_nodes_);
	EXPECT_EQ(cg.in_nodes_, cg2.in_nodes_);
	EXPECT_EQ(cg.out_nodes_, cg2.out_nodes_);
	EXPECT_EQ(cg.internal_nodes_, cg2.internal_nodes_);


	EXPECT_EQ(cg, cg2);
}

}

