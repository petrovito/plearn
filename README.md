# plearn-AD

Plearn is an *algorithmic differentiation* library. It was created as a personal project in order to gain a deeper understanding for what is happening behind the scenes with machine learning libraries such as *tensorflow*, and not as an effort to try to with compete with them. (*To see if I could not learn what it had to teach.*)

## Example code 

	// create model  with:
	//    * 1 input of shape {3}
	//    * 1 variable of shape {3}
	//    * 1 output of shape {1}
	Model m;
	auto input = m.add_input({3});
	auto variable = m.add_variable({3});
	
	auto difference = input - variable;
	auto sq = difference.square();
	auto reduce = sq.reduce_sum(0);

	m.set_output(reduce);
	m.compile();
	
	// initialize the variable
	variable.set_tensor(Tensors::create({3}, new float[]{1, 2, 3}));
	// create model-input
	auto input_ten = Tensors::create({3}, new float[]{2,4,6});
	// execute the model given the variable and input tensors
	auto output = m.execute({input_ten}, true);

	auto out_data = output.tensor_of(reduce)->data(); //buffer of out tensor (of size 1)
	ASSERT_EQ(out_data[0], 1 + 4 + 9);

	// get derivative of `reduce` wrt to `variable` (a tensor of shape{3,1})
	auto& var_out_grad = output.grad_of(variable, reduce);
	ASSERT_EQ(var_out_grad.in_shape, shape_t{3});
	ASSERT_EQ(var_out_grad.out_shape, shape_t{1});
	auto out_diffs = var_out_grad.data(); // buffer of derivative
	ASSERT_EQ(out_diffs[0], -2);
	ASSERT_EQ(out_diffs[1], -4);
	ASSERT_EQ(out_diffs[2], -6);

