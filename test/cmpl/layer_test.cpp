#include <gtest/gtest.h>

#include <cmpl/layers.h>
#include <cmpl/model.h>

using namespace plearn;

TEST(Model, Basic) {
	
	Input<S<64, 10>> input;
	Output<S<64, 2>> output;
	auto layer1 = DenseLayer(S<64,10>{}, S<64, 100>{});
	auto layer2 = DenseLayer(S<64,100>{}, S<64, 32>{});
	auto layer3 = DenseLayer(S<64,32>{}, S<64, 1>{});
	auto m = Model(input);
	auto m1 = m << layer1;

	auto m2 = m1 << layer2;
	auto m3 = m2 << layer3;
	auto m3_out = m3.output_;
	{
		std::cout<<"M3 out: ";
		for (int i = 0; i < m3.output_.shape.rank; i++) 
			std:: cout << m3.output_.shape.dims[i]<<", ";
		std::cout <<std::endl;
	}

	Tensor data(input.shape);
	auto prediction = m3.predict(data);
}

