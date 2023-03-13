#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <operation.h>
#include <vector>

#include <cpu_types.h>
#include <cpu_ops.h>
#include <call_graph.h>

namespace plearn {

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
					auto cpu_tens = cpu_tensor_factory::allocate(tens);
					env_->tensors_.push_back(cpu_tens);
					flow_n.tensor_ = cpu_tens;
				} 
				return *this;
			}

			cpu_exec_env_builder& load_data_nodes(hash_map<node_id, cpu_tensor> cpu_tensors) {
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

		static unique_ptr<cpu_exec_env> create_env(call_graph& graph) {
		}
		
		static vector<cpu_tensor> execute(const vector<cpu_tensor>& input, cpu_exec_env& env) {
			auto x = env.reset(input);
			while (env.state() == env_state::IN_PROGRESS) {
				//TODO propagate...
			}
			//TODO
			return {};
		}

		static void execute_op(operation op, vector<cpu_tensor> inputs, cpu_tensor& output) {
			switch (op.type_) {
				case op_type::matmul:
					cpu_matmul(op, inputs, output);
				case op_type::matvecmul:
					cpu_matvecmul(op, inputs, output);
				case op_type::add:
					cpu_add(op, inputs, output);
			}

		}

	};

}

