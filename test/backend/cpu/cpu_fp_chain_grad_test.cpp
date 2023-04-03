#include <gtest/gtest.h>

#include "backend/cpu/cpu_types.h"
#include "backend/cpu/cpu_backend.h"
#include <backend/cpu/cpu_fp_grad.h>

namespace plearn::backend::cpu {


cpu_backend backend = cpu_backend{};

TEST(CpuFpChainGrad, VecMatMul) {
	operation op = vecmatmul{};	
	/* tensor_p v = backend.create_tensor(shape_t{3}); */
}

}

