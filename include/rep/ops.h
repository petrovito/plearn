#pragma once

#include "rep/rep_types.h"
#include <compare>

namespace plearn::rep {


	enum class op_type {
		noop,
		identity,
		matmul,
		vecmatmul,
		sub,
		add,
		mult,
		square,
	};

	/**
	 * Instruction on how to modify output tensor
	 */
	enum class output_modify_t {
		set,
		add,
		substract,
	};

	struct operation {
		op_type type_;
		friend auto operator<=>(const operation&, const operation&) = default;
	};

	struct noop : public operation {
		noop() : operation{op_type::noop} { }
	};

	struct matmul : public operation {
		matmul() : operation{op_type::matmul} {}
	};

	struct vecmatmul : public operation {
		vecmatmul() : operation{op_type::vecmatmul} {}
	};

	struct add : public operation {
		add() : operation{op_type::add} {}
	};
	
	struct sub : public operation {
		sub() : operation{op_type::sub} {}
	};

	struct mult : public operation {
		mult() : operation{op_type::mult} {}
	};

	struct square : public operation {
		square() : operation{op_type::square} {}
	};


}

