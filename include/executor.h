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

	template<typename T>
	using read_ptr = const T*;

	template<typename T>
	using borrowed_ptr = T*;

	template<typename T>
	using owned_ptr = T*;


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

	using std::shared_ptr;

	class cpu_tensor {
		public:
			cpu_tensor() = default;
			cpu_tensor(const tensor& tens, const shared_ptr<tensor_buf>& buf) :
				meta_data_{tens}, content_{buf} {}
			borrowed_ptr<tensor_buf> get_content() const {
				return content_.get();
			}

		private:
			tensor meta_data_;
			shared_ptr<tensor_buf> content_;

		friend class cpu_tensor_factory;
	};


	class cpu_tensor_factory {
		public:
			owned_ptr<cpu_tensor> allocate(const tensor& tens) const {
				auto buf = std::make_shared<tensor_buf>(tens.shape_.size());
				return new cpu_tensor(tens, buf);
			}
	} constexpr cpu_tensor_fac;

	struct cpu_op_node;

	struct cpu_tensor_node {
		node_id id_;
		shape shape_;
		cpu_tensor* tensor_;
		vector<cpu_op_node*> outputs_;
	};


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

		//owned tensors
		vector<unique_ptr<cpu_tensor>> tensors_;
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

				for (auto in_node_id: graph.in_nodes_) {
					auto& in_node = env_->tensor_nodes_[in_node_id];
					env_->in_nodes_.push_back(&in_node);
				}
				//TODO outnodes?
			}

			cpu_exec_env_builder& alloc_flow_mem() {
				for (auto flown_id: flow_nodes_) {
					auto& flow_n = env_->tensor_nodes_[flown_id];
					tensor tens(flow_n.shape_);
					auto cpu_tens = cpu_tensor_fac.allocate(tens);
					env_->tensors_.push_back(unique_ptr<cpu_tensor>(cpu_tens));
					flow_n.tensor_ = cpu_tens;
				} 
				return *this;
			}

			cpu_exec_env_builder& load_data_nodes(hash_map<node_id, borrowed_ptr<cpu_tensor>> cpu_tensors) {
				for (auto& [n_id, cpu_tens]: cpu_tensors) {
					env_->tensor_nodes_[n_id].tensor_ = cpu_tens;
				}
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

