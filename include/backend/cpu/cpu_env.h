#pragma once

#include "backend/cpu/cpu_types.h"
#include "backend/cpu/cpu_ops.h"

namespace plearn::backend::cpu {

	struct cpu_op_node;

	struct cpu_tensor_node {
		node_id id_;
		shape_t shape_;
		cpu_tensor tensor_{};
		vector<cpu_op_node*> outputs_{};

		void set_cpu_tensor(const cpu_tensor& cpu_tens) {
			assert(cpu_tens.meta_data().shape() == shape_);
			tensor_ = cpu_tens;
		}
	};



	struct cpu_op_node {
		struct dep {
			node_id id_;
			cpu_tensor_node* ten_node_{nullptr};
			bool is_ready_{false};
			bool is_flow_node_{false};
		};

		node_id id_;
		operation op_;

		vector<dep> deps_{};
		int unready_deps_{0};
		int flow_dep_count_{0};
		cpu_tensor_node* out_{nullptr};

		//returns true IFF this op_node just become ready as a result of this dep
		bool dep_ready(node_id id) {
			auto dep = find_if(deps_, [id](auto& dep) {return id == dep.id_;});
			if (dep == deps_.end() && dep->is_ready_) return false;
			//not ready yet
			dep->is_ready_ = true;
			return unready_deps_ == 0;
		}
	};

	class cpu_exec_env {
		public:
			/** 
			 *  Zeros out flow tensors, and resets dependency counters.
			 *  Sets input nodes of the call graph.
			 * */
			void reset(const vector<cpu_tensor>& input) {
				assert(state_ == run_state::READY);
				assert(input.size() == in_nodes_.size());
				state_ = run_state::IN_PROGRESS;
				unready_out_tens_ = out_nodes_.size();
				//zero flow tensors
				for (auto flown: flow_nodes_) {
					flown->tensor_.zero();
				}
				//reset dependency counters and readiness for flow nodes
				for (auto& [id, opn]: op_nodes_) {
					opn.unready_deps_ = opn.flow_dep_count_;
					for (auto dep: opn.deps_) {
						if (dep.is_flow_node_) dep.is_ready_ = false;
					}
				}
				//set inputs
				for (uint64_t i = 0; i < input.size(); i++) {
					in_nodes_[i]->set_cpu_tensor(input[i]);
					for (auto opn: in_nodes_[i]->outputs_) {
						set_dep_ready(opn, in_nodes_[i]);
					}
				}
			}

			cpu_op_node* pop_ready_op() {
				assert(!ready_q_.empty());
				auto front = ready_q_.front();
				ready_q_.pop();
				return front;
			}

			void flow_node_ready(cpu_tensor_node* flown) {
				for (auto opn: flown->outputs_) {
					set_dep_ready(opn, flown);
				}
				if (std::ranges::count(out_nodes_, flown) > 0) {
					unready_out_tens_--;
					if (unready_out_tens_ == 0) {
						state_ = run_state::READY;
					}
				}
			}

			vector<cpu_tensor> output_tensors() {
				assert(state_==run_state::READY);
				vector<cpu_tensor> outputs(out_nodes_.size());
				std::transform(out_nodes_.begin(), out_nodes_.end(), outputs.begin(),
						[] (auto outn) { return outn->tensor_; });
				return outputs;
			}


			run_state state() const { return state_; }

			const tensor& tensor_at(node_id id) const {
				return tensor_nodes_.at(id).tensor_.meta_data();
			}

		private:
			//returns true IFF op_node just become ready as a result of this dep
			bool set_dep_ready(cpu_op_node* opn, const cpu_tensor_node* inn) {
				if (opn->dep_ready(inn->id_)) {
					ready_q_.push(opn);
					return true;
				} 
				return false;
			}


			run_state state_;
			queue<cpu_op_node*> ready_q_;
			int unready_out_tens_;
			

			hash_map<node_id, cpu_tensor_node> tensor_nodes_;
			hash_map<node_id, cpu_op_node> op_nodes_;

			vector<cpu_tensor_node*> in_nodes_;
			vector<cpu_tensor_node*> out_nodes_;
			vector<cpu_tensor_node*> flow_nodes_;

		friend class cpu_exec_env_builder;
		friend class CpuExecutor_ExecEnvBuilder_Test;
		friend class CpuExecutor_ExecEnvExecute_Test;
	};


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
					tensor tens = tensor_factory::create(flow_n.shape_);
					flow_n.tensor_ = cpu_tensor_factory::allocate(tens);				} 
				return *this;
			}

			cpu_exec_env_builder& load_data_nodes(const hash_map<node_id, cpu_tensor>& data_tensors) {
				for (auto& [n_id, cpu_tens]: data_tensors) {
					env_->tensor_nodes_[n_id].tensor_ = cpu_tens;
				}
				return *this;
			}

			unique_ptr<cpu_exec_env> build() {
				env_->state_ = run_state::READY;
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
		public:

		static unique_ptr<cpu_exec_env> create_env(call_graph& graph) {
			return {};
		}
		
		static vector<cpu_tensor> execute(const vector<cpu_tensor>& input, cpu_exec_env& env) {
			env.reset(input);
			while (env.state() == run_state::IN_PROGRESS) {
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
				case op_type::vecmatmul:
					cpu_vecmatmul(op, inputs, output);
					break;
				case op_type::add:
					cpu_add(op, inputs, output);
					break;
			}

		}

	};


}
