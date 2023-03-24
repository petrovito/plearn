#pragma once

#include "cmpl/tensor.h"
#include "cmpl/layers.h"

namespace plearn {



	template<typename _ShapeIn, typename _ShapeOut=_ShapeIn,
		typename... _Layers>
	class Model {
		public:
			using TOut = Tensor<_ShapeOut>;
			using TIn = Tensor<_ShapeIn>;

			constexpr static Input<_ShapeIn> input_{};
			constexpr static Output<_ShapeOut> output_{};
			constexpr static GenericSeq<_Layers...> layers_{};

			Model(Input<_ShapeIn>) {}

			Model(Input<_ShapeIn>, Output<_ShapeOut>, GenericSeq<_Layers...>) {}

			Tensor<_ShapeOut> predict(TIn) {
				return Tensor(output_.shape);
			}
			

	};

	template<typename _ShapeIn, typename _ShapeOut, typename _ShapeLayerOut, 
		template <typename...> class LayerT,
		typename... _Layers>
	requires LayerConcept<LayerT<_ShapeIn, _ShapeLayerOut>, _ShapeIn, _ShapeLayerOut> 
			auto operator<<(Model<_ShapeIn, _ShapeOut, _Layers...>& model,
					LayerT<_ShapeOut, _ShapeLayerOut>& layer) {
				auto new_layers = pack(model.layers_, layer);
				return Model(model.input_, layer.output_, new_layers);
			}

}
