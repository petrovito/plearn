#pragma once

#include <cstdint>

#include <memory>
#include <rep/rep_types.h>

namespace plearn::env {

	using namespace rep;

	using tensor_id = uint64_t;
	using std::unique_ptr;
	using std::shared_ptr;
	

	class exec_env;
	class env_section;
	class backend_t;
	class tensor_back_t;


	class tensor_t {
		public:
			shape_t shape() const { return shape_; }
			tensor_id id() const { return id_; }
			borrowed_ptr<tensor_back_t> back() { return back_.get(); }
		private:
			tensor_t(const shape_t& s, tensor_id id, 
					unique_ptr<tensor_back_t>&& ten_b) : 
				shape_{s}, id_{id}, back_{std::move(ten_b)} {}

			shape_t shape_;
			tensor_id id_;
			unique_ptr<tensor_back_t> back_;
			borrowed_ptr<exec_env> env_;
		friend class tensor_factory;
	};

	using tensor_p = shared_ptr<tensor_t>;
	
	class tensor_factory {
		public:
			tensor_p create(const shape_t& s, unique_ptr<tensor_back_t>&& back) {
				return tensor_p(new tensor_t{s, next_id(), std::move(back)});
			}
		private:
			tensor_id next_id() { return id++; }

			tensor_id id = 1;
	};


	struct exec_params {
		hash_map<node_id, tensor_p> inputs_;
		hash_map<node_id, tensor_p> outputs_;
	};

	struct exec_result {
		bool success{true};
	};


	class backend_t {
		public:
			virtual void exec_op(const operation& op, 
					const vector<tensor_p>& inputs, tensor_p output) = 0;
			virtual unique_ptr<tensor_back_t> create_tensor(const shape_t& s) = 0;
			virtual ~backend_t() = default;
	};

	class tensor_back_t {
		public:
			virtual ~tensor_back_t() = default;
	};

}
