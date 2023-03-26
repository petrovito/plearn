#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include <ranges>
#include <vector>

#include <rep/rep_types.h>
#include <rep/call_graph.h>
#include <rep/forward_prop.h>
#include <environ/env_types.h>
#include <environ/env_section.h>

namespace plearn::env {


	/**
	 * An exec_env is a container for sections, allocated resources (fx. memory),
	 * and a backend executor.
	 */
	class exec_env {
		public:
			const backend_t& backend() const { return *backend_; }

			env_section create_section() { return env_section{this}; }

			virtual tensor_p create_tensor(const shape_t& s) {
				auto ten_b = backend_->create_tensor(s);
				return tens_fac_.create(s, std::move(ten_b));
			}
		protected:
			exec_env() {}

			tensor_factory tens_fac_{};
			unique_ptr<backend_t> backend_;
		private:

			friend class EnvSection_Execute_Test;
	};

}

