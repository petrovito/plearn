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



	class Model {

		class ModelTensorT {
			public:
				node_id id() const { return id_; }

			private:
				ModelTensorT(const shape_t& shape, Model& model, node_id id) 
					: shape_(shape), model_(model), id_(id) {}

				const shape_t shape_;
				Model& model_;
				const node_id id_;

				//is data node or flow node
				bool internal_ = false;

				friend class Model;
				friend class ModelTensor;
		};


		class ModelTensor : private shared_ptr<ModelTensorT> {
			public:
				ModelTensorT* get() const { return shared_ptr<ModelTensorT>::get(); }
				ModelTensorT* operator->() const { return shared_ptr<ModelTensorT>::operator->(); }

				[[nodiscard]]
				static ModelTensor create(const shape_t& shape, Model& model, node_id id) {
					return {new ModelTensorT(shape, model, id)};
				}

				[[nodiscard]]
					ModelTensor operator+(const ModelTensor& other) const {
						if (get()->shape_ != other.get()->shape_) 
							throw std::runtime_error("Shape mismatch");
						return get()->model_.add_operation(add{}, get()->shape_, *this, other);
					}
				[[nodiscard]]
					ModelTensor operator-(const ModelTensor& other) const {
						if (get()->shape_ != other.get()->shape_) 
							throw std::runtime_error("Shape mismatch");
						return get()->model_.add_operation(sub{}, get()->shape_, *this, other);
					}

				[[nodiscard]]
					ModelTensor operator*(const ModelTensor& other) const {
						if (get()->shape_ != other.get()->shape_) 
							throw std::runtime_error("Shape mismatch");
						return get()->model_.add_operation(mult{}, get()->shape_, *this, other);
					}

				[[nodiscard]]
					ModelTensor matmul(const ModelTensor& other) const {
						const shape_t& shape = get()->shape_;
						if (shape.rank != 2 && other->shape_.rank != 2 &&
								shape.dims[1] != other->shape_.dims[0])
							throw std::runtime_error("Shape mismatch");
						shape_t out_shape{shape.dims[0], other->shape_.dims[1]};
						return get()->model_.add_operation(rep::matmul{}, out_shape, *this, other);
					}

				[[nodiscard]]
					ModelTensor dot_product(const ModelTensor& other) const {
						const shape_t& shape = get()->shape_;
						if (shape.rank != 1 && other->shape_.rank != 1 &&
								shape.dims[0] != other->shape_.dims[0])
							throw std::runtime_error("Shape mismatch");
						shape_t out_shape{1};
						return get()->model_.add_operation(rep::dot_product{}, out_shape, *this, other);
					}

				[[nodiscard]]
					ModelTensor vecmatmul(const ModelTensor& other) const {
						const shape_t& shape = get()->shape_;
						if (shape.rank != 1 && other->shape_.rank != 2 &&
								shape.dims[0] != other->shape_.dims[0])
							throw std::runtime_error("Shape mismatch");
						shape_t out_shape{other->shape_.dims[1]};
						return get()->model_.add_operation(rep::vecmatmul{}, out_shape, *this, other);
					}

				[[nodiscard]]
					ModelTensor square() const {
						return get()->model_.add_operation(rep::square{}, get()->shape_, *this);
					}

				void set_tensor(tensor_p t) { get()->model_.set_variable_tensor(*this, t); }

			private:
				ModelTensor(ModelTensorT* t) : shared_ptr<ModelTensorT>(t) {}
		};

		public:
			Model() :
				exec_env_(ExecEnvProvider::get_exec_env()) { }
			
			[[nodiscard]]
			ModelTensor add_variable(shape_t shape) {
				auto nid = cg_builder_.add_data_node(shape);
				auto& tensor = tensors_.emplace_back(ModelTensor::create(shape, *this, nid));
				variables_.push_back(tensor);
				tensors_map_[nid] = &tensor;
				return tensor;
			}

			[[nodiscard]]
			ModelTensor add_input(shape_t shape) {
				auto nid = cg_builder_.add_input_node(shape);
				auto& tensor = tensors_.emplace_back(ModelTensor::create(shape, *this, nid));
				inputs_.push_back(tensor);
				tensors_map_[nid] = &tensor;
				return tensor;
			}

			template <typename ...Input>
			[[nodiscard]]
			ModelTensor add_operation(const operation& op, const shape_t& output_shape,
					Input&&... inputs) {
				vector<node_id> input_ids;
				((input_ids.push_back(inputs->id_), ...));
				auto [_, nid] = cg_builder_.add_op_node(op, input_ids, output_shape);
				auto& tensor = tensors_.emplace_back(ModelTensor::create(output_shape, *this, nid));
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


			void set_variable_tensor(ModelTensor& m_tensor, tensor_p t) {
				env_section_->set_data_tensor(m_tensor->id_, t);
			}



			vector<tensor_p> execute(const vector<tensor_p>& inputs) {
				unordered_map<node_id, tensor_p> input_tensors;
				for (unsigned idx = 0; idx < inputs_.size(); idx++) {
					auto& t = inputs_[idx];
					input_tensors[t->id_] = inputs[idx]; 
				}
				unordered_map<node_id, tensor_p> output_tensors;
				for (auto& t : outputs_)
					output_tensors[t->id_] = exec_env_->create_tensor(t->shape_, tensor_init::zero);
				exec_params params{.inputs_=input_tensors, .outputs_=output_tensors};
				env_section_->execute(params);
				vector<tensor_p> outputs;
				for (unsigned idx = 0; idx < outputs_.size(); idx++) {
					auto& t = outputs_[idx];
					outputs.push_back(params.outputs_[t->id_]);
				}
				return outputs;
			}

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
