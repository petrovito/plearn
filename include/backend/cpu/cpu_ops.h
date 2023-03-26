#pragma once

#include <rep/call_graph.h>

#include <backend/cpu/cpu_types.h>
#include <backend/cpu/cpu_op_impl.h>

namespace plearn::backend::cpu {

	inline void cpu_matmul(operation, const vector<cpu_tensor*>& inputs, cpu_tensor* output) {
		auto mat1 = inputs[0]->get_content()->buf;		
		auto mat2 = inputs[1]->get_content()->buf;		
		auto mat_out = output->get_content()->buf;
		auto shape1 = inputs[0]->shape();
		auto shape2 = inputs[1]->shape();
		_cpu_matmul(mat1, mat2, mat_out, shape1.dims[shape1.rank-2], 
				shape1.dims[shape1.rank-1], shape2.dims[shape2.rank -1]);
	}

	inline void cpu_vecmatmul(operation, const vector<cpu_tensor*>& inputs, cpu_tensor*& output) {
		auto vec = inputs[0]->get_content()->buf;		
		auto mat2 = inputs[1]->get_content()->buf;		
		auto mat_out = output->get_content()->buf;
		auto shape2 = inputs[1]->shape();
		_cpu_vecmatmul(vec, mat2, mat_out, shape2.dims[shape2.rank-2], 
				shape2.dims[shape2.rank-1]);
	}

	inline void cpu_matvecmul(operation, const vector<cpu_tensor*>& inputs, cpu_tensor*& output) {
		auto mat1 = inputs[0]->get_content()->buf;		
		auto mat2 = inputs[1]->get_content()->buf;		
		auto mat_out = output->get_content()->buf;
		auto shape1 = inputs[0]->shape();
		_cpu_matvecmul(mat1, mat2, mat_out, shape1.dims[shape1.rank-2], 
				shape1.dims[shape1.rank-1]);
	}

	inline void cpu_add(operation, const vector<cpu_tensor*>& inputs, cpu_tensor*& output) {
		auto mat1 = inputs[0]->get_content()->buf;		
		auto mat2 = inputs[1]->get_content()->buf;		
		auto mat_out = output->get_content()->buf;
		auto size = inputs[0]->shape().size();
		_cpu_add(mat1, mat2, mat_out, size);
	}

}

