#pragma once

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
		output_modify_t modify_ = output_modify_t::set;
		
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


}

