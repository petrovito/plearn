#pragma once

#include <model/model.h>

namespace plearn::model {



	class DenseLayer {
		public:
			[[nodiscard]]
			DenseLayer(Model::ModelTensor input, int output_size) 
				: model_(input->model()), input_(input), output_size_(output_size) {
					model_.unset_output(input);
					A_ = model_.add_variable({input->shape()[0], output_size});
					b_ = model_.add_variable({output_size});
					flow1_ = input_.vecmatmul(A_);
					flow2_ = flow1_ + b_;
					model_.set_output(flow2_);
					model_.commit();
				}

			Model::ModelTensor output() const { return flow2_; }

			void set_tensors(tensor_p A, tensor_p b) {
				A_.set_tensor(A);
				b_.set_tensor(b);
			}

			Model::ModelTensor A() const { return A_; }
			Model::ModelTensor b() const { return b_; }


		private:
			Model& model_;
			Model::ModelTensor input_;

			Model::ModelTensor A_;
			Model::ModelTensor b_;

			Model::ModelTensor flow1_, flow2_;

			int output_size_;
	};


}

