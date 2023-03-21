#pragma once

#include <algorithm>
#include <bits/ranges_util.h>
#include "call_graph.h"
#include <cassert>
#include <cstdint>
#include <unordered_set>
#include <ranges>
#include <queue>
#include <vector>

namespace plearn {

	using std::unique_ptr;
	using std::shared_ptr;

	template<typename T>
	using read_ptr = const T*;

	template<typename T>
	using borrowed_ptr = T*;

	template<typename T>
	using owned_ptr = T*;


	struct tensor_buf {
		float* buf;
		uint64_t size;

		tensor_buf(uint64_t size) : size(size),
			buf{new (std::align_val_t(32)) float[size]{}} { }
		~tensor_buf() { delete [] buf; }
	};


	class cpu_tensor {
		public:
			cpu_tensor() = default;
			cpu_tensor(const tensor& tens, const shared_ptr<tensor_buf>& buf) :
				meta_data_{tens}, content_{buf} {}

			borrowed_ptr<tensor_buf> get_content() const { return content_.get(); }
			const tensor& meta_data() const { return meta_data_; }

			/**
			 *  Zero out tensor content.
			 */
			void zero() { std::fill_n(content_->buf, meta_data_.shape_.size(), 0); }

		private:
			tensor meta_data_;
			shared_ptr<tensor_buf> content_;

		friend class cpu_tensor_factory;
	};


	class cpu_tensor_factory {
		public:
			static cpu_tensor allocate(const tensor& tens) {
				auto buf = std::make_shared<tensor_buf>(tens.shape_.size());
				return cpu_tensor(tens, buf);
			}
	};

	struct cpu_op_node;

	struct cpu_tensor_node {
		node_id id_;
		shape shape_;
		cpu_tensor tensor_{};
		vector<cpu_op_node*> outputs_{};

		void set_cpu_tensor(const cpu_tensor& cpu_tens) {
			assert(cpu_tens.meta_data().shape_ == shape_);
			tensor_ = cpu_tens;
		}
	};

	using std::ranges::find_if;


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

	enum class env_state {
		IN_PROGRESS,
		READY,
	};

	using std::queue;
	using std::unordered_set;

	class cpu_exec_env {
		public:
			/** 
			 *  Zeros out flow tensors, and resets dependency counters.
			 *  Sets input nodes of the call graph.
			 * */
			void reset(const vector<cpu_tensor>& input) {
				assert(state_ == env_state::READY);
				assert(input.size() == in_nodes_.size());
				state_ = env_state::IN_PROGRESS;
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
						state_ = env_state::READY;
					}
				}
			}

			vector<cpu_tensor> output_tensors() {
				assert(state_==env_state::READY);
				vector<cpu_tensor> outputs(out_nodes_.size());
				std::transform(out_nodes_.begin(), out_nodes_.end(), outputs.begin(),
						[] (auto outn) { return outn->tensor_; });
				return outputs;
			}


			env_state state() const { return state_; }

		private:
			//returns true IFF op_node just become ready as a result of this dep
			bool set_dep_ready(cpu_op_node* opn, const cpu_tensor_node* inn) {
				if (opn->dep_ready(inn->id_)) {
					ready_q_.push(opn);
					return true;
				} 
				return false;
			}


			env_state state_;
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
	
}

