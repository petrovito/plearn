#pragma once

#include <algorithm>
#include <bits/ranges_util.h>
#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <ranges>
#include <queue>
#include <vector>

#include "rep/call_graph.h"
#include "environ/env_types.h"

namespace plearn::backend::cpu {

	using namespace plearn::rep;
	using namespace plearn::env;

	using std::unique_ptr;
	using std::shared_ptr;
	using std::queue;
	using std::unordered_set;
	using std::ranges::find_if;

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
			void zero() { std::fill_n(content_->buf, meta_data_.shape().size(), 0); }

		private:
			tensor meta_data_;
			shared_ptr<tensor_buf> content_;

		friend class cpu_tensor_factory;
	};


	class cpu_tensor_factory {
		public:
			static cpu_tensor allocate(const shape_t& shape) {
				return allocate(tensor_factory::create(shape));
			}

			static cpu_tensor allocate(const tensor& tens) {
				auto buf = std::make_shared<tensor_buf>(tens.shape().size());
				return cpu_tensor(tens, buf);
			}
	};


	
}

