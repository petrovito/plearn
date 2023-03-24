#pragma once

#include "rep/call_graph.h"

#include "backend/cpu/cpu_types.h"
#include "cpu_op_impl.h"

namespace plearn {

	inline void cpu_matmul(operation, vector<cpu_tensor> inputs, cpu_tensor& output) {
		auto mat1 = inputs[0].get_content()->buf;		
		auto mat2 = inputs[1].get_content()->buf;		
		auto mat_out = output.get_content()->buf;
		auto shape1 = inputs[0].meta_data().shape();
		auto shape2 = inputs[1].meta_data().shape();
		_cpu_matmul(mat1, mat2, mat_out, shape1.dims[shape1.rank-2], 
				shape1.dims[shape1.rank-1], shape2.dims[shape2.rank -1]);
	}

	inline void cpu_matvecmul(operation, vector<cpu_tensor> inputs, cpu_tensor& output) {
		auto mat1 = inputs[0].get_content()->buf;		
		auto mat2 = inputs[1].get_content()->buf;		
		auto mat_out = output.get_content()->buf;
		auto shape1 = inputs[0].meta_data().shape();
		_cpu_matvecmul(mat1, mat2, mat_out, shape1.dims[shape1.rank-2], 
				shape1.dims[shape1.rank-1]);
	}

	inline void cpu_add(operation, vector<cpu_tensor> inputs, cpu_tensor& output) {
		auto mat1 = inputs[0].get_content()->buf;		
		auto mat2 = inputs[1].get_content()->buf;		
		auto mat_out = output.get_content()->buf;
		auto size = inputs[0].meta_data().shape().size();
		_cpu_add(mat1, mat2, mat_out, size);
	}

}

