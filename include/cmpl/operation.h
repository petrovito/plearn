#pragma once

#include <array>
#include <cstdint>
#include "tensor.h"
#include <type_traits>
#include <concepts>
#include <utility>

namespace plearn {

	template <typename sh1, typename sh2,  typename shOut>
	concept Operation = 
		requires (Tensor<sh1> a, Tensor<sh2> b){
			{ apply(a, b) } -> std::convertible_to<Tensor<shOut>>;
	};

	template<typename __Shape>
	class Add {
		using TT = Tensor<__Shape>;

		public:
			Add(TT& a, TT&b) : a(a), b(b) {};
			Tensor<__Shape> apply() {
				return {};
			}
		private:
			TT& a;
			TT& b;
	};

	template<typename __ShapeA, typename __ShapeB>
	constexpr bool matmul_compatible() {
		int rankA = __ShapeA::rank;
		if (rankA < 2) return false;
		if (rankA != __ShapeB::rank) return false;
		for (int i = 0; i < __ShapeA::rank -2; i++) {
			if (__ShapeA::dims[i] != __ShapeB::dims[i]) return false;
		}
		return __ShapeA::dims[rankA -1] == __ShapeB::dims[rankA -2];
	}



	template<typename __ShapeA, typename __ShapeB>
	requires (matmul_compatible<__ShapeA, __ShapeB>())
	class MatMul {
		using TA = Tensor<__ShapeA>;
		using TB = Tensor<__ShapeB>;

		public:
			MatMul(TA& a, TB&b) : a(a), b(b) {};
			auto apply() {
				auto all_but_last = allButLast(__ShapeA::seq);
				const auto last = lastElem(__ShapeB::seq);
				const auto new_shape_s = pack<last>(all_but_last);
				const auto new_shape = shapeOf(new_shape_s);
				return tensorOf(new_shape);
			}
		private:
			TA& a;
			TB& b;
	};
}

