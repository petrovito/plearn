#pragma once

#include <cstdint>

#include <rep/rep_types.h>

namespace plearn::env {

	using namespace rep;

	using tensor_id = uint64_t;
	

	class exec_env;
	class section;
	class backend;
	class tensor_back;


	class tensor_t {
		public:
			shape_t shape() const { return shape_; }
			tensor_id id() const { return id_; }
		private:
			tensor_t(const shape_t& s, tensor_id id, 
					unique_ptr<tensor_back>&& ten_b) : 
				shape_{s}, id_{id}, back_{std::move(ten_b)} {}

			shape_t shape_;
			tensor_id id_;
			unique_ptr<tensor_back> back_;
			borrowed_ptr<exec_env> env_;
		friend class tensor_factory;
	};

	using tensor_p = borrowed_ptr<tensor_t>;
	
	class tensor_factory {
		public:
			tensor_t create(const shape_t& s, unique_ptr<tensor_back>&& back) {
				return tensor_t{s, next_id(), std::move(back)};
			}
		private:
			tensor_id next_id() { return id++; }

			tensor_id id = 1;
	};


	struct exec_params {
		vector<tensor_p> inputs;
	};

	struct exec_result {
		vector<tensor_p> outputs;
	};


	class backend {
		public:
			virtual exec_result execute(const exec_params& params) = 0;
			virtual unique_ptr<tensor_back> create_tensor(const shape_t& s) = 0;
			virtual ~backend() = default;
	};

}
