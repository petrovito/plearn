#pragma once

#include <algorithm>
#include <bits/ranges_algo.h>
#include <cstdint>
#include <memory>
#include "operation.h"
#include <ranges>
#include <vector>

#include "cpu_types.h"
#include "cpu_ops.h"
#include "call_graph.h"


namespace plearn {


	/**
	 * A gradient tensor of tensor WRT. an input tensor.
	 * (For some underlying function)
	 * Identity and independentness (null-tensor) are not stored as a cpu_tensor, but
	 * just a boolean field.
	 */
	struct cpu_gradient {
		cpu_gradient(bool identity) : 
			grad_{}, identity_(identity), independent_(false) {};
		cpu_gradient(bool identity, bool independent) :
			grad_{}, identity_(identity), independent_(independent) {};
		cpu_tensor grad_;
		bool identity_;
		bool independent_;
	};

	const cpu_gradient independent_gradient = cpu_gradient(false, true);


	/**
	 * A derivative system is a set of gradients for a given tensor.
	 * The gradients are indexed by the tensor id of the input tensor.
	 * The gradients are stored in a hash map.
	 */
	struct cpu_derivative_system {
		cpu_derivative_system() : grads{} {};
		
		static cpu_derivative_system identity(tensor_id id) {
			cpu_derivative_system d;
			d.grads.insert({id, cpu_gradient(true)});
			return d;
		}

		const cpu_gradient& operator[](tensor_id id) {
			if (grads.contains(id)) return grads[id];
			return independent_gradient;
		}

		hash_map<tensor_id, cpu_gradient> grads;
	};


	class cpu_forward_prop_diff {

		public:
			cpu_forward_prop_diff() :
				data_tensors_{}, derivatives_{}
		{};

		private:
			vector<tensor> data_tensors_;
			hash_map<tensor_id, cpu_derivative_system> derivatives_;
			friend class cpu_forward_prop_diff_builder;


	};
	

	class cpu_forward_prop_diff_builder {

		public:
			cpu_forward_prop_diff_builder(const call_graph& graph) :
				diff_env_{std::make_unique<cpu_forward_prop_diff>()},
				graph_{graph} { }
			
			/**
			 * The nodes for which the derivatives will be computed.
			 */
			void data_nodes(const vector<tensor>& data_nodes) {
				data_tensors_ = data_nodes;
				diff_env_->data_tensors_ = data_nodes;
			}

			/**
			 * Find the nodes that are dependant on the requested data nodes.
			 */
			void find_dependant_nodes() {
				if (data_tensors_.empty() || !dependant_nodes_.empty()) return;
				for (auto& data_tensor: data_tensors_) {
					derivatives_.insert({data_tensor.id(),
							cpu_derivative_system::identity(data_tensor.id())});
				}

			}

			unique_ptr<cpu_forward_prop_diff> build() {
				return std::move(diff_env_);
			}


		private:
			unique_ptr<cpu_forward_prop_diff> diff_env_;
			const call_graph& graph_;
			vector<tensor> data_tensors_;
			vector<tensor_id> dependant_nodes_;
			hash_map<tensor_id, cpu_derivative_system> derivatives_;
	};
	
}

