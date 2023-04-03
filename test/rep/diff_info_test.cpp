#include <gtest/gtest.h>
#include <unordered_set>

#include <rep/call_graph.h>
#include <rep/diff_info.h>

using namespace plearn::rep;

TEST(ForwardProp, Builder) {
	call_graph_builder cg_builder;
	auto inn_id = cg_builder.add_input_node(shape_t{768});
	auto data1n_id = cg_builder.add_data_node(shape_t{768, 128});
	auto [op1n_id, flown_id] = 
		cg_builder.add_op_node(matmul{}, {inn_id, data1n_id}, shape_t{128});
	auto data2n_id = cg_builder.add_data_node(shape_t{128, 10});
	auto [op2n_id, outn_id] = 
		cg_builder.add_op_node(matmul{}, {flown_id, data2n_id}, shape_t{10});
	cg_builder.make_output(outn_id);
	auto cg = cg_builder.build();

	diff_info_builder deps_builder(cg);
	auto cg_deps = deps_builder.all_data_nodes()
		.find_dependencies()
		.build();
	
	auto datan_set = unordered_set({data1n_id, data2n_id});
	auto fp_variables = unordered_set<node_id>(
			cg_deps->variable_nodes().begin(), 
			cg_deps->variable_nodes().end());
	ASSERT_EQ(datan_set, fp_variables);
	
	auto deps = cg_deps->dependencies();
	// there is a derivative entry for all tensor nodes
	// 1 input, 2 data, 2 flow (1 flow+ 1 output)
	ASSERT_EQ(deps.size(), 1 + 2 + 2);
	// all diffs have entries for all data nodes
	for (auto& [id, diff]: deps) {
		for (auto datan_id: datan_set) {
			ASSERT_TRUE(diff.variable_deps_.contains(datan_id));
		}
	}
	
	// the derivative for the input tensor is independent
	ASSERT_FALSE(deps[inn_id].depends_on_any());

	// data nodes only depend on themselves
	unordered_set<node_id> data1n_deps = {data1n_id};
	ASSERT_EQ(deps[data1n_id].variable_dependencies(), data1n_deps);
	unordered_set<node_id> data1n_direct_deps = {data1n_id};
	ASSERT_EQ(deps[data1n_id].op_input_deps_.size(), data1n_direct_deps.size());
	for (auto nid: data1n_direct_deps) {
		ASSERT_TRUE(deps[data1n_id].op_input_deps_.contains(nid));
	}
	unordered_set<node_id> data2n_deps = {data2n_id};
	ASSERT_EQ(deps[data2n_id].variable_dependencies(), data2n_deps);
	unordered_set<node_id> data2n_direct_deps = {data2n_id};
	ASSERT_EQ(deps[data2n_id].op_input_deps_.size(), data2n_direct_deps.size());
	for (auto nid: data2n_direct_deps) {
		ASSERT_TRUE(deps[data2n_id].op_input_deps_.contains(nid));
	}

	// the flow tensor depends on the first data tensor
	unordered_set<node_id> flown_deps = {data1n_id};
	ASSERT_EQ(deps[flown_id].variable_dependencies(), flown_deps);
	unordered_set<node_id> flown_direct_deps = {inn_id, data1n_id};
	ASSERT_EQ(deps[flown_id].op_input_deps_.size(), flown_direct_deps.size());
	for (auto nid: flown_direct_deps) {
		ASSERT_TRUE(deps[flown_id].op_input_deps_.contains(nid));
	}

	// the output tensor depends on both data tensors
	unordered_set<node_id> outn_deps = {data1n_id, data2n_id};
	ASSERT_EQ(deps[outn_id].variable_dependencies(), outn_deps);
	unordered_set<node_id> outn_direct_deps = {flown_id, data2n_id};
	ASSERT_EQ(deps[outn_id].op_input_deps_.size(), outn_direct_deps.size());
	for (auto nid: outn_direct_deps) {
		ASSERT_TRUE(deps[outn_id].op_input_deps_.contains(nid));
	}

	//everything is a dependency for output
	for (auto& [datan_id, _]: cg.data_nodes_) {
		ASSERT_TRUE(deps[datan_id].output_dependant(outn_id));
	}
	for (auto& [flown_id, _]: cg.flow_nodes_) {
		ASSERT_TRUE(deps[flown_id].output_dependant(outn_id));
	}
}

