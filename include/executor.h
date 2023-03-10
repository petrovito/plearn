#pragma once

#include <cstdint>
#include <memory>
#include <operation.h>
#include <vector>

#include <call_graph.h>

namespace plearn {

	using std::unique_ptr;

	class exec_env {};

	/* template<typename T> */
	/* using owned_ptr = T*; */


	class Executor {
		virtual unique_ptr<exec_env> create_env(call_graph&) = 0;
		virtual vector<tensor> execute(vector<tensor> input, exec_env&) = 0;
	};


	struct tensor_buf {
		float* buf;
		uint64_t size;

		tensor_buf(uint64_t size) : size(size),
			buf{new (std::align_val_t(64)) float[size]{}} { }
		~tensor_buf() { delete [] buf; }
	};

	class cpu_tensor {
		public:
			tensor meta_data_;
			std::shared_ptr<tensor_buf> content_;

			cpu_tensor(const shape& shape) : meta_data_(shape) {
				std::size_t size = 1;
				for (auto dim : shape.dims) {
					size *= dim;
				}
				content_ = std::make_shared<tensor_buf>(size);
			}

			tensor_buf* get_content() const {
				return content_.get();
			}
	};

	struct op_node;

	struct cpu_tensor_node {
		node_id id_;
		cpu_tensor* tensor_;
	};

	/* struct cpu_placeholder_node { */
	/* 	node_id id_; */
	/* 	shape shape_; */
	/* 	vector<op_node*> outputs_; */
	/* }; */


	struct cpu_op_node {
		node_id id_;
		operation op_;

		vector<cpu_tensor_node*> ints_;
		int unready_deps_;
		cpu_tensor_node* out_;
	};

	struct cpu_exec_env {
		/* hash_map<node_id, cpu_placeholder_node> in_nodes_; */
		/* hash_map<node_id, cpu_tensor_node> int_nodes_; */
		hash_map<node_id, cpu_tensor_node> tensor_nodes_;
		hash_map<node_id, cpu_op_node> op_nodes_;

		vector<cpu_tensor_node*> in_nodes_;
	};


	class CpuExecutor {

		unique_ptr<cpu_exec_env> create_env(call_graph& graph) {
			auto env = new cpu_exec_env{};
			//create nodes
			for (auto& [id, node]: graph.flow_nodes_) {
				env->tensor_nodes_.insert({id, {.id_=id}});
				//TODO allocate memory
			}
			for (auto& [id, node]: graph.data_nodes_) {
				env->tensor_nodes_.insert({id, {.id_=id}});
				//TODO load tensor into memory... how?
			}
			for (auto& [id, node]: graph.op_nodes_) {
				env->op_nodes_.insert({id, {.id_=id}});
			}
			//TODO wiring
			return unique_ptr<cpu_exec_env>(env);
		}
		
		vector<tensor> execute(vector<tensor> input, exec_env&) {
		}

	};

}

