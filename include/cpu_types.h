#pragma once

#include <algorithm>
#include <call_graph.h>
#include <cassert>
#include <cstdint>
#include <unordered_set>

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
		cpu_tensor tensor_;
		vector<cpu_op_node*> outputs_;

		void set_cpu_tensor(const cpu_tensor& cpu_tens) {
			assert(cpu_tens.meta_data().shape_ == shape_);
			tensor_ = cpu_tens;
		}
	};


	struct cpu_op_node {
		node_id id_;
		operation op_;

		vector<cpu_tensor_node*> deps_;
		int unready_deps_;
		int flow_dep_count_;
		cpu_tensor_node* out_;
	};

	enum class env_state {
		IN_PROGRESS,
		READY,
	};

	class cpu_exec_env {
		public:
			/** 
			 *  Zeros out flow tensors, and resets dependency counters.
			 *  Sets input nodes of the call graph.
			 * */
			auto reset(const vector<cpu_tensor>& input) {
				assert(state_ == env_state::READY);
				assert(input.size() == in_nodes_.size());
				state_ = env_state::IN_PROGRESS;
				//zero flow tensors
				for (auto flown: flow_nodes_) {
					flown->tensor_.zero();
				}
				//reset dependency counters
				for (auto& [id, opn]: op_nodes_) {
					opn.unready_deps_ = opn.flow_dep_count_;
				}
				std::unordered_set<cpu_op_node*> ops_in_sight;
				//set inputs
				for (uint64_t i = 0; i < input.size(); i++) {
					in_nodes_[i]->set_cpu_tensor(input[i]);
					for (auto opn: in_nodes_[i]->outputs_) {
						ops_in_sight.emplace(opn);
					}
				}
				return ops_in_sight;
			}

			env_state state() const { return state_; }

		private:
			env_state state_;

			hash_map<node_id, cpu_tensor_node> tensor_nodes_;
			hash_map<node_id, cpu_op_node> op_nodes_;

			vector<cpu_tensor_node*> in_nodes_;
			vector<cpu_tensor_node*> flow_nodes_;

			//owned tensors
			vector<cpu_tensor> tensors_;

		friend class cpu_exec_env_builder;
	};
	
}

