#include <gtest/gtest.h>
#include "call_graph_test_helper.h"
#include "persistence.h"
#include <call_graph.h>
#include <string>


namespace plearn {

TEST(Persistence, SaveLoad) {
	std::string path = "/tmp/test.pb";
	
	call_graph cg = call_graph_example();
	Persistence::save(cg, path);
	call_graph cg2 = Persistence::load(path);
	EXPECT_EQ(cg, cg2);
}

}

