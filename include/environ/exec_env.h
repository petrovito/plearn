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

namespace plearn::env {


	/**
	 * A section contains a set of representations and necessary run time variables
	 * required to execute a call graph, and calculate derivatives.
	 *
	 * The section belongs to an execution environment.
	 */
	class env_section {
		public:
			exec_result execute(const exec_params& params);
			

		private:
			env_section(borrowed_ptr<exec_env> env) : env_{env} {}

			call_graph cg_;
			unique_ptr<forward_prop_diff> fp_diff_;
			borrowed_ptr<exec_env> env_;

			friend class exec_env;
	};


	/**
	 * An exec_env is a container for sections, allocated resources (fx. memory),
	 * and a backend executor.
	 */
	class exec_env {
		public:
			static borrowed_ptr<exec_env> default_env() {
				static exec_env env{};
				return &env;
			}
			env_section create_section() {
				return env_section{this};
			}

			tensor_t create_tensor(const shape_t& s) {
				auto ten_b = backend_->create_tensor(s);
				return tens_fac_.create(s, std::move(ten_b));
			}
		private:
			exec_env() {}

			tensor_factory tens_fac_{};
			unique_ptr<backend> backend_;
	};

}

