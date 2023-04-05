#pragma once

#include "environ/exec_env.h"
#include "rep/call_graph.h"
#include "rep/ops.h"
#include "rep/rep_types.h"
#include <environ/env_types.h>
#include <environ/env_section.h>
#include <memory>
#include <model/exec_env_provider.h>
#include <unordered_map>
#include <vector>

namespace plearn::model {
	using namespace ::plearn::env;


	class Model;
	
	class ModelTensorT {
		public:
			ModelTensorT(const shape_t& shape, Model& model, node_id id) 
				: shape_(shape), model_(model), id_(id) {}

			node_id id() const { return id_; }

		private:
			const shape_t shape_;
			const Model& model_;
			const node_id id_;

			//is data node or flow node
			bool internal_ = false;

			friend class Model;
	};

	using ModelTensor = shared_ptr<ModelTensorT>;


	class Model {
		public:
			Model() :
				exec_env_(ExecEnvProvider::get_exec_env()) { }
			
			[[nodiscard]]
			ModelTensor add_variable(shape_t shape) {
				auto nid = cg_builder_.add_data_node(shape);
				auto& tensor = tensors_.emplace_back(std::make_shared<ModelTensorT>(shape, *this, nid));
				variables_.push_back(tensor);
				tensors_map_[nid] = &tensor;
				return tensor;
			}

			[[nodiscard]]
			ModelTensor add_input(shape_t shape) {
				auto nid = cg_builder_.add_input_node(shape);
				auto& tensor = tensors_.emplace_back(std::make_shared<ModelTensorT>(shape, *this, nid));
				inputs_.push_back(tensor);
				tensors_map_[nid] = &tensor;
				return tensor;
			}

			template <typename ...Input>
			[[nodiscard]]
			ModelTensor add_operation(const operation& op, const shape_t& output_shape,
					Input&&... inputs) {
				vector<node_id> input_ids(sizeof...(inputs));
				((input_ids.push_back(inputs->id_), ...));
				auto [_, nid] = cg_builder_.add_op_node(op, input_ids, output_shape);
				auto& tensor = tensors_.emplace_back(std::make_shared<ModelTensorT>(output_shape, *this, nid));
				flow_tensors_.push_back(tensor);
				tensors_map_[nid] = &tensor;
				return tensor;
			}

			void set_output(ModelTensor& tensor) {
				cg_builder_.make_output(tensor->id_);
				outputs_.push_back(tensor);
			}

			void compile() {
				cg_ = cg_builder_.build();
				env_section_builder section_builder(exec_env_, exec_env_->backend(), cg_);
				env_section_ = section_builder.allocate_internal_tensors().build();
			}

/* 			vector<ModelTensor> execute(const vector<ModelTensor> inputs) { */
/* 				unordered_map<node_id, tensor_p> input_tensors; */
/* 				for (auto& t : inputs) */
/* 					input_tensors[t->id_] = t->tensor_; */
/* 				unordered_map<node_id, tensor_p> output_tensors; */
/* 				for (auto& t : outputs_) */
/* 					output_tensors[t->id_] = exec_env_->create_tensor(t->shape_); */
/* 				exec_params params{.inputs_ = input_tensors, .outputs_ = output_tensors}; */
/* 				env_section_->execute(params); */
/* 				vector<ModelTensor> outputs; */
/* 				for (auto& t : outputs_) { */
/* 					auto& tensor = outputs.emplace_back(std::make_shared<ModelTensorT>(t->shape_, *this, t->id_)); */
/* 					tensor->tensor_ = output_tensors[t->id_]; */
/* 				} */
/* 				return outputs; */
/* 			} */

		private:
			vector<ModelTensor> tensors_;
			vector<ModelTensor> variables_;
			vector<ModelTensor> flow_tensors_;
			vector<ModelTensor> inputs_;
			vector<ModelTensor> outputs_;
			unordered_map<node_id, ModelTensor*> tensors_map_;

			borrowed_ptr<exec_env> exec_env_;
			unique_ptr<env_section> env_section_;

			call_graph cg_;
			call_graph_builder cg_builder_;


	};



}
