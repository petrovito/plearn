#pragma once

#include <cstdint>

#include "rep/rep_types.h"

namespace plearn::env {

	using namespace rep;

	using tensor_id = uint64_t;


	class tensor {
		public:
			tensor() = default;
			shape_t shape() const { return shape_; }
			tensor_id id() const { return id_; }
		private:
			tensor(const shape_t& s, tensor_id id) : shape_{s}, id_{id} {}
			shape_t shape_;
			tensor_id id_;
		friend class tensor_factory;

	};
	
	class tensor_factory {
		public:
			static tensor create(const shape_t& s) {
				return tensor{s, next_id()};
			}
		private:
			static tensor_id next_id() {
				static tensor_id id = 1;
				return id++;
			}
	};

}
