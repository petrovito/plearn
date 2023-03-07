#pragma once

#include <cstdint>
#include <operation.h>
#include <vector>

#include <call_graph.h>

namespace plearn {


	class exec_env {};


	class Executor {
		virtual exec_env create_env(call_graph&) = 0;
		virtual vector<tensor> execute(vector<tensor> input, exec_env&) = 0;
	};


	class cpu_tensor {
		public:
			tensor data_;
			std::shared_ptr<float[]> content_;

			cpu_tensor(const shape& shape) : data_(shape) {
				std::size_t size = 1;
				for (auto dim : shape.dims) {
					size *= dim;
				}
				content_ = std::make_shared<float[]>(size);
			}

			float* get_content() const {
				return content_.get();
			}
	};

	class cpu_exec_env {
		vector<cpu_tensor> in_nodes_;
		vector<cpu_tensor> int_nodes_;
		vector<cpu_tensor> out_nodes_;
		vector<cpu_tensor> data_nodes_;
		vector<cpu_tensor> op_nodes_;
	};


	class CpuExecutor : public Executor {

	};

}


