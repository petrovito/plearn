#pragma once

#include "environ/exec_env.h"
#include "rep/rep_types.h"
#include <backend/cpu/cpu_backend.h>

namespace plearn::model {
	using namespace ::plearn::env;

	class ExecEnvProvider {
		public:
			static borrowed_ptr<exec_env> get_exec_env() {
				static backend::cpu::cpu_backend backend_ = backend::cpu::cpu_backend();
				static exec_env exec_env_ = exec_env(&backend_);
				return &exec_env_;
			}
			
		private:
	};


	class Tensors {
		public:
			Tensors() : exec_env_{ExecEnvProvider::get_exec_env()} {}

			[[nodiscard]]
			static tensor_p create(shape_t shape) {
				static Tensors tensors_;
				return tensors_.exec_env_->create_tensor(shape);
			}

			[[nodiscard]]
			static tensor_p create(shape_t shape, float* data) {
				static Tensors tensors_;
				return tensors_.exec_env_->create_tensor(shape, data);
			}
		private:
			borrowed_ptr<exec_env> exec_env_;
	};

}
