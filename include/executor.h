#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include <operation.h>
#include <ranges>
#include <vector>

#include <cpu_types.h>
#include <cpu_ops.h>
#include <call_graph.h>

namespace plearn {

	class cpu_exec_env_builder {

		public:
			cpu_exec_env_builder(const call_graph& graph) :
				env_{std::make_unique<cpu_exec_env>()},
				in_nodes_{graph.in_nodes_}, 
				out_nodes_{graph.out_nodes_}
			{
				//create nodes
				for (auto& [id, node]: graph.flow_nodes_) {
					env_->tensor_nodes_.insert({id, {.id_=id, .shape_=node.shape_}});
					flow_nodes_.push_back(id);
				}
				for (auto& [id, node]: graph.data_nodes_) {
					env_->tensor_nodes_.insert(
							{id, {.id_=id, .shape_=node.shape_}});
					data_nodes_.push_back(id);
				}
				for (auto& [id, node]: graph.op_nodes_) {
					env_->op_nodes_.insert({id, {.id_=id, .op_=node.op_}});
				}

				//wiring
				for (auto& [opn_id, op_node]: graph.op_nodes_) {
					auto& cpu_op_node = env_->op_nodes_[opn_id];
					for (auto tensorn_id: op_node.inputs_) {
						bool is_flow_node = graph.flow_nodes_.contains(tensorn_id);
						auto& cpu_tensor_node = env_->tensor_nodes_[tensorn_id];
						cpu_op_node.deps_.push_back({
								.id_=cpu_tensor_node.id_, .ten_node_=&cpu_tensor_node, 
								.is_ready_=true, .is_flow_node_=is_flow_node});
						cpu_tensor_node.outputs_.push_back(&cpu_op_node);
					}
					cpu_op_node.out_ = &env_->tensor_nodes_[op_node.out_];
				}

				for (auto in_node_id: in_nodes_) {
					auto& in_node = env_->tensor_nodes_[in_node_id];
					env_->in_nodes_.push_back(&in_node);
				}
				for (auto out_node_id: out_nodes_) {
					auto& out_node = env_->tensor_nodes_[out_node_id];
					env_->out_nodes_.push_back(&out_node);
				}
				for (auto flow_node_id: flow_nodes_) {
					auto& flow_node = env_->tensor_nodes_[flow_node_id];
					env_->flow_nodes_.push_back(&flow_node);
				}
			}

			cpu_exec_env_builder& alloc_flow_mem() {
				//TODO except for input nodes
				for (auto flown_id: flow_nodes_) {
					auto& flow_n = env_->tensor_nodes_[flown_id];
					tensor tens(flow_n.shape_);
					flow_n.tensor_ = cpu_tensor_factory::allocate(tens);				} 
				return *this;
			}

			cpu_exec_env_builder& load_data_nodes(hash_map<node_id, cpu_tensor> data_tensors) {
				for (auto& [n_id, cpu_tens]: data_tensors) {
					env_->tensor_nodes_[n_id].tensor_ = cpu_tens;
				}
				return *this;
			}

			unique_ptr<cpu_exec_env> build() {
				env_->state_ = env_state::READY;
				return std::move(env_); 
			}

		private:
			unique_ptr<cpu_exec_env> env_;
			vector<node_id> flow_nodes_,
							data_nodes_,
							in_nodes_,
							out_nodes_;
	}; 


	class CpuExecutor {

		static unique_ptr<cpu_exec_env> create_env(call_graph& graph) {
			return {};
		}
		
		static vector<cpu_tensor> execute(const vector<cpu_tensor>& input, cpu_exec_env& env) {
			env.reset(input);
			while (env.state() == env_state::IN_PROGRESS) {
				auto opn = env.pop_ready_op();
				vector<cpu_tensor> input_tens(opn->deps_.size());
				std::transform(opn->deps_.begin(), opn->deps_.end(), input_tens.begin(),
						[](auto dep) {return dep.ten_node_->tensor_;});
				execute_op(opn->op_, input_tens, opn->out_->tensor_);
				env.flow_node_ready(opn->out_);
			}
			return env.output_tensors();
		}

		static void execute_op(operation op, vector<cpu_tensor> inputs, cpu_tensor& output) {
			switch (op.type_) {
				case op_type::noop:
					break;
				case op_type::matmul:
					cpu_matmul(op, inputs, output);
					break;
				case op_type::matvecmul:
					cpu_matvecmul(op, inputs, output);
					break;
				case op_type::add:
					cpu_add(op, inputs, output);
					break;
			}

		}

	};

}

