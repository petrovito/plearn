#include <gtest/gtest.h>
#include <string>

#include "../rep/call_graph_test_helper.h"

#include "data/persistence.h"
#include "rep/call_graph.h"

namespace plearn {

TEST(Persistence, SaveLoad) {
	std::string path = "/tmp/test.pb";
	
	call_graph cg = call_graph_example();
	Persistence::save(cg, path);
	call_graph cg2 = Persistence::load(path);
	EXPECT_EQ(cg, cg2);
}

}

