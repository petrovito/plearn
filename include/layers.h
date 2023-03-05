#pragma once

#include <concepts>
#include <operation.h>

namespace plearn {

	template<typename _Shape> 
	struct Input {
		constexpr static _Shape shape{};
	};

	template<typename _Shape> 
	struct Output {
		constexpr static _Shape shape{};
	};

	template<typename _Layer, typename _ShapeIn, typename _ShapeOut>
		concept LayerConcept = requires (_Layer l, _ShapeIn sIn, _ShapeOut sOut) {
			{ _Layer(sIn, sOut) };
			{ _Layer(Input<_ShapeIn>{}, Output<_ShapeOut>{}) };
		} 
		&& std::same_as<decltype(_Layer::input_), const Input<_ShapeIn>>
		&& std::same_as<decltype(_Layer::output_), const Output<_ShapeOut>>;

	template<typename _ShapeIn, typename _ShapeOut>
	class DenseLayer {
		public: 
			constexpr static Input<_ShapeIn> input_{};
			constexpr static Output<_ShapeOut> output_{};
			
			DenseLayer(_ShapeIn, _ShapeOut) {}
			DenseLayer(Input<_ShapeIn>, Output<_ShapeOut>) {}
	};
}

