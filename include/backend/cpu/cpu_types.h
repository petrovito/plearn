#pragma once

#include "rep/rep_types.h"
#include <algorithm>
#include <bits/ranges_util.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <ranges>
#include <queue>
#include <vector>

#include <rep/call_graph.h>
#include <environ/env_types.h>

namespace plearn::backend::cpu {

	using namespace plearn::rep;
	using namespace plearn::env;

	using std::unique_ptr;
	using std::shared_ptr;
	using std::queue;
	using std::unordered_set;
	using std::ranges::find_if;


	struct tensor_buf {
		float* buf;
		uint64_t size;

		tensor_buf(uint64_t size) : size(size),
			buf{new (std::align_val_t(32)) float[size]{}} { }
		~tensor_buf() { delete [] buf; }
	};


	class cpu_tensor : public tensor_back_t {
		public:
			borrowed_ptr<tensor_buf> get_content() const { return content_.get(); }
			const shape_t& shape() const { return shape_; }

			/**
			 *  Zero out tensor content.
			 */
			void zero() { std::fill_n(content_->buf, shape_.size(), 0); }

		private:
			cpu_tensor(shape_t shape, const shared_ptr<tensor_buf>& buf) :
				shape_{shape}, content_{buf} {}

			shape_t shape_;
			shared_ptr<tensor_buf> content_;

		friend class cpu_tensor_factory;
	};


	class cpu_tensor_factory {
		public:
			static cpu_tensor allocate(const shape_t& shape) {
				auto buf = std::make_shared<tensor_buf>(shape.size());
				return cpu_tensor(shape, buf);
			}
	};


	
}

