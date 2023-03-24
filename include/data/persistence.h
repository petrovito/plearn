#pragma once

#include <fstream>

#include <data/converter.h>

namespace plearn::data {

	class Persistence {
		public:
			static void save(const call_graph& graph, const std::string& path) {
				CallGraphM* graph_m = GraphConverter::to_proto(graph);
				std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
				if (!graph_m->SerializeToOstream(&output)) {
					std::cerr << "Failed to write graph." << std::endl;
					exit(1);
				}
				delete graph_m;
			}

			static call_graph load(const std::string& path) {
				CallGraphM graph_m;
				std::fstream input(path, std::ios::in | std::ios::binary);
				if (!graph_m.ParseFromIstream(&input)) {
					std::cerr << "Failed to parse graph." << std::endl;
					exit(1);
				}
				return GraphConverter::to_call_graph(graph_m);
			}
	};

}
