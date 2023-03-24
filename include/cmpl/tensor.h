#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <memory>

#include <cmpl/meta_helpers.h>

namespace plearn {

	using tdata_t = float;
	using std::array;

	template<uint64_t... dim>
	struct S {
		static constexpr uint64_t size() {
			uint64_t size = 1;
			for (uint64_t d: {dim...}) size *= d;
			return size;
		}

		static constexpr auto seq = IntSeq<dim...>{};


		static constexpr uint8_t rank = sizeof...(dim);
		static constexpr std::array<uint64_t, rank> dims = {dim...};

		static constexpr S<dim...> inst() { return {}; };
	};


	template <typename __Shape>
	class Tensor {
		public:
			static constexpr auto shape = __Shape::inst();

			using tarray = array<tdata_t, shape.size()>;

			Tensor() {}
			Tensor(__Shape) {}


		private:
	};


	//helper functions

	template<uint64_t... dims>
		constexpr S<dims...> shapeOf(IntSeq<dims...>) { return {}; }

	template<typename __Shape>
		constexpr Tensor<__Shape> tensorOf(__Shape) { return {}; };

}

