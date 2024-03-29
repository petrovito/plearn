#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/diff_info.h>
#include <environ/env_types.h>

namespace plearn::env {


	/**
	 * An exec_env is a container for sections, allocated resources (fx. memory),
	 * and a backend executor.
	 */
	class exec_env {
		public:
			exec_env(backend_t* backend) : backend_{backend} {}

			[[nodiscard]]
			borrowed_ptr<backend_t> backend() const { return backend_; }

			[[nodiscard]]
			virtual tensor_p create_tensor(const shape_t& s, 
					tensor_init init=tensor_init::no_init) {
				auto ten_b = backend_->create_tensor(s, init);
				return tens_fac_.create(s, std::move(ten_b));
			}

			[[nodiscard]]
			virtual tensor_p create_tensor(const shape_t& s, float* data) {
				auto ten_b = backend_->create_tensor(s, data);
				return tens_fac_.create(s, std::move(ten_b));
			}
		protected:

			tensor_factory tens_fac_{};
			borrowed_ptr<backend_t> backend_;
		private:

			friend class EnvSection_Execute_Test;
	};

}

