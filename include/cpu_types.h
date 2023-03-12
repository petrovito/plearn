#pragma once

#include <call_graph.h>

namespace plearn {

	using std::unique_ptr;
	using std::shared_ptr;

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
			buf{new (std::align_val_t(32)) float[size]{}} { }
		~tensor_buf() { delete [] buf; }
	};


	class cpu_tensor {
		public:
			cpu_tensor() = default;
			cpu_tensor(const tensor& tens, const shared_ptr<tensor_buf>& buf) :
				meta_data_{tens}, content_{buf} {}

			borrowed_ptr<tensor_buf> get_content() const { return content_.get(); }
			const tensor& meta_data() { return meta_data_; }

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
		vector<cpu_tensor> tensors_;
	};
	
}

