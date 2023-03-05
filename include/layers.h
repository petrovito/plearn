#pragma once

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

	template<typename _ShapeIn, typename _ShapeOut>
	class Layer {
		public: 
			constexpr static Input<_ShapeIn> input_{};
			constexpr static Output<_ShapeOut> output_{};
			
			Layer(_ShapeIn, _ShapeOut) {}
			Layer(Input<_ShapeIn>, Output<_ShapeOut>) {}
	};

	template<typename _ShapeIn, typename _ShapeOut=_ShapeIn,
		typename... _Layers>
	class Model {
		public:

			constexpr static Input<_ShapeIn> input_{};
			constexpr static Output<_ShapeOut> output_{};
			constexpr static GenericSeq<_Layers...> layers_{};

			Model(Input<_ShapeIn>) {}

			Model(Input<_ShapeIn>, Output<_ShapeOut>, GenericSeq<_Layers...>) {}

	};

	template<typename _ShapeIn, typename _ShapeOut,
		typename _ShapeLayerOut,
		typename... _Layers>
			auto operator<<(Model<_ShapeIn, _ShapeOut, _Layers...>& model,
					Layer<_ShapeOut, _ShapeLayerOut>& layer) {
				auto new_layers = pack(model.layers_, layer);
				return Model(model.input_, layer.output_, new_layers);
			}

}

