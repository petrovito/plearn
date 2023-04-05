#pragma once

#include <cstdint>

#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <rep/rep_types.h>
#include "rep/call_graph.h"

namespace plearn::env {

	using namespace rep;

	using tensor_id = uint64_t;
	using std::unique_ptr;
	using std::shared_ptr;
	using std::unordered_map;
	

	class exec_env;
	class env_section;
	class backend_t;

	class tensor_back_t {
		public:
			virtual float* data() = 0;
			virtual void zero() = 0; //anull tensor
			virtual ~tensor_back_t() = default;
	};


	class tensor_t {
		public:
			const shape_t& shape() const { return shape_; }
			tensor_id id() const { return id_; }
			borrowed_ptr<tensor_back_t> back() { return back_.get(); }
			borrowed_ptr<float> data() { return back_->data(); }
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


	/*struct grad_wrt { */
	/*	node_id input; */
	/*	node_id output; */

	/*	bool operator==(const grad_wrt& other) const = default; */
	/*}; */

	/*/1** */
	/* * Hashing function for grad_wrt */
	/* *1/ */
	/*struct grad_wrt_hash { */
	/*	std::size_t operator()(const grad_wrt& gw) const { */
	/*		return std::hash<node_id>{}(gw.input) ^ std::hash<node_id>{}(gw.output); */
	/*	} */
	/*}; */

	class gradient {
		public:
			shape_t in_shape;
			shape_t out_shape;
			
			/* bool identity{false}; */

			shared_ptr<tensor_back_t> back_{};
	};

	struct node_grad {
		node_id in_id_;
		node_id out_id_;
		gradient grad_;
		bool identity_{false};
	};
	
	using grad_map = unordered_map<node_id, node_grad>;
	const grad_map empty_grad_map{};

	struct grad_system : public unordered_map<node_id, grad_map> {

		const grad_map& at(node_id id) const {
			if (contains(id))
				return unordered_map<node_id, grad_map>::at(id);
			return empty_grad_map;
		}

	};


	struct exec_params {
		bool calc_diffs{false};

		hash_map<node_id, tensor_p> inputs_{};
		hash_map<node_id, tensor_p> outputs_{};
	};

	struct exec_result {
		bool success_{true};
		borrowed_ptr<grad_system> grad_system_{nullptr};
	};



	class op_exec_backend_t {
		public:
			virtual void exec_op(const operation& op, 
					const vector<tensor_p>& inputs, tensor_p& output) = 0;

			virtual ~op_exec_backend_t() = default;
	};


	class fw_op_diff_backend_t {
		public:
			virtual void reset(const vector<tensor_p>& inputs, const tensor_p& output) {
				this->inputs_ = &inputs;
				this->output_ = &output;
			}
			virtual void update_grad_with_identity(unsigned input_idx, 
					gradient& var_out_grad) { 
				(void)input_idx;
				(void)var_out_grad;
				throw std::runtime_error("unimplemented"); 
			}
			virtual void update_grad(unsigned input_idx, 
					const gradient& var_in_grad, gradient& var_out_grad) = 0;
		
		protected:
			read_ptr<vector<tensor_p>> inputs_;
			read_ptr<tensor_p> output_;
	};

	class bw_op_diff_backend_t {
		public:
			virtual void reset(const vector<tensor_p>& inputs, const tensor_p& output) {
				this->inputs_ = &inputs;
				this->output_ = &output;
			}
			/* virtual void update_grad_with_identity(unsigned input_idx, */ 
			/* 		gradient& var_out_grad) { */ 
			/* 	(void)input_idx; */
			/* 	(void)var_out_grad; */
			/* 	throw std::runtime_error("unimplemented"); */ 
			/* } */
			virtual void update_grad(unsigned input_idx, 
					const gradient& out_outn_grad, gradient& in_outn_grad) = 0;
			virtual ~bw_op_diff_backend_t() = default;
		
		protected:
			read_ptr<vector<tensor_p>> inputs_;
			read_ptr<tensor_p> output_;
	};

	class bp_diff_backend_t {
		public:
			[[nodiscard]]
			virtual unique_ptr<bw_op_diff_backend_t> create_op_bw_diff_backend(const operation& op) = 0;

			virtual ~bp_diff_backend_t() = default;
	};

	class fp_diff_backend_t {
		public:
			[[nodiscard]]
			virtual unique_ptr<fw_op_diff_backend_t> create_op_fw_diff_backend(const operation& op) = 0;

			virtual ~fp_diff_backend_t() = default;
	};

	enum class tensor_init {
		no_init,
		zero,
		identity
	};

	class backend_t : public op_exec_backend_t, public fp_diff_backend_t, public bp_diff_backend_t {
		public:
			[[nodiscard]]
			virtual unique_ptr<tensor_back_t> create_tensor(const shape_t& s,
					tensor_init init=tensor_init::no_init) = 0;

			[[nodiscard]]
			virtual unique_ptr<tensor_back_t> create_tensor(const shape_t& s, float* data) = 0;

			virtual ~backend_t() = default;
	};



	/**
	 * Container for all the tensors that are required for call graph executions.
	 */
	struct section_exec_tensors {
		template<typename... Args>
		section_exec_tensors(
				hash_map<node_id, tensor_p>& tensors,
				Args&&... other_tens

				) :
			tensors_{tensors} {
				(tensors_.insert(other_tens.begin(), other_tens.end()), ...);

			}
		hash_map<node_id, tensor_p> tensors_;

		tensor_p& operator[](node_id id) { return tensors_[id]; }
	};

	class diff_env {
		public:
			virtual void calc_diff(const op_node& opn, const vector<tensor_p>& inputs, const tensor_p& output) = 0;
			virtual void calc_diffs(section_exec_tensors&) = 0;
			virtual void reset() = 0;
			virtual borrowed_ptr<grad_system> get_grad_system() = 0;
			virtual ~diff_env() = default;
	};


}
