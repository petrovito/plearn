#pragma once

#include <algorithm>
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

	struct cpu_op_node;

	struct cpu_tensor_node {
		node_id id_;
		shape shape_;
		cpu_tensor* tensor_;
		vector<cpu_op_node*> outputs_;
	};

	/* struct cpu_placeholder_node { */
	/* 	node_id id_; */
	/* 	shape shape_; */
	/* 	vector<op_node*> outputs_; */
	/* }; */


	struct cpu_op_node {
		node_id id_;
		operation op_;

		vector<cpu_tensor_node*> deps_;
		int unready_deps_;
		cpu_tensor_node* out_;
	};

	struct cpu_exec_env {
		hash_map<node_id, cpu_tensor_node> tensor_nodes_;
		hash_map<node_id, cpu_op_node> op_nodes_;

		vector<cpu_tensor_node*> in_nodes_;
	};
	

	class cpu_exec_env_builder {

		public:
			cpu_exec_env_builder(const call_graph& graph) :
				env_{std::make_unique<cpu_exec_env>()}
			{
				//create nodes
				for (auto& [id, node]: graph.flow_nodes_) {
					env_->tensor_nodes_.insert({id, {.id_=id, .shape_=node.shape_}});
					flow_nodes_.push_back(id);
				}
				for (auto& [id, node]: graph.data_nodes_) {
					env_->tensor_nodes_.insert(
							{id, {.id_=id, .shape_=node.tensor_->shape_}});
					data_nodes_.push_back(id);
				}
				for (auto& [id, node]: graph.op_nodes_) {
					env_->op_nodes_.insert({id, {.id_=id}});
				}

				//wiring
				for (auto& [opn_id, op_node]: graph.op_nodes_) {
					auto& cpu_op_node = env_->op_nodes_[opn_id];
					for (auto tensorn_id: op_node.inputs_) {
						auto& cpu_tensor_node = env_->tensor_nodes_[tensorn_id];
						cpu_op_node.deps_.push_back(&cpu_tensor_node);
						cpu_tensor_node.outputs_.push_back(&cpu_op_node);
					}
				}
			}

			cpu_exec_env_builder& alloc_flow_mem() {
				return *this;
			}

			cpu_exec_env_builder& load_data_nodes() {
				return *this;
			}

			unique_ptr<cpu_exec_env> build() { return std::move(env_); }

		private:
			unique_ptr<cpu_exec_env> env_;
			vector<node_id> flow_nodes_,
							data_nodes_,
							in_nodes_,
							out_nodes_;
	}; 


	class CpuExecutor {

		unique_ptr<cpu_exec_env> create_env(call_graph& graph) {
		}
		
		vector<tensor> execute(vector<tensor> input, exec_env&) {
		}

	};

}

