#pragma once

#include <cstdint>

#include <memory>
#include <rep/rep_types.h>
#include <unordered_map>

namespace plearn::env {

	using namespace rep;

	using tensor_id = uint64_t;
	using std::unique_ptr;
	using std::shared_ptr;
	using std::unordered_map;
	

	class exec_env;
	class env_section;
	class backend_t;
	class tensor_back_t;


	class tensor_t {
		public:
			const shape_t& shape() const { return shape_; }
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
			
			bool identity{false};

			shared_ptr<tensor_back_t> back_;
	};
	
	using grad_map = unordered_map<node_id, gradient>;
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
		bool success{true};
		borrowed_ptr<grad_system> grads;
	};


	class backend_t {
		public:
			virtual void exec_op(const operation& op, 
					const vector<tensor_p>& inputs, tensor_p& output) = 0;
			
			virtual void calc_forward_grad(const operation& op,
					const vector<tensor_p>& inputs, const tensor_p& output,
					const vector<read_ptr<grad_map>>& in_grads, grad_map& out_grad) = 0;

			virtual unique_ptr<tensor_back_t> create_tensor(const shape_t& s) = 0;
			virtual ~backend_t() = default;
	};

	class tensor_back_t {
		public:
			virtual void zero() = 0; //anull tensor
			virtual ~tensor_back_t() = default;
	};

}
