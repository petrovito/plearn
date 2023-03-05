#include <gtest/gtest.h>
#include <layers.h>

using namespace plearn;

TEST(Model, Basic) {
	
	Input<S<64, 10>> input;
	Output<S<64, 2>> output;
	auto layer1 = Layer(S<64,10>{}, S<64, 100>{});
	auto layer2 = Layer(S<64,100>{}, S<64, 32>{});
	auto layer3 = Layer(S<64,32>{}, S<64, 1>{});
	auto m = Model(input);
	auto m1 = m << layer1;
	auto m2 = m1 << layer2;
	auto m3 = m2 << layer3;
	std::cout<<"M3 out: ";
		for (int i = 0; i < m3.output_.shape.rank; i++) 
    std:: cout << m3.output_.shape.dims[i]<<", ";
	std::cout <<std::endl;

}

