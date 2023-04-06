#pragma once

#include "rep/rep_types.h"
#include <compare>

namespace plearn::rep {


	enum class op_type {
		noop,
		identity,
		matmul,
		vecmatmul,
		matvecmul,
		dot_product,
		sub,
		add,
		mult,
		square,

		reduce_sum,
		reduce_mean,
	};


	struct operation {
		op_type type_;
		int iarg0_{};
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

	struct matvecmul : public operation {
		matvecmul() : operation{op_type::matvecmul} {}
	};

	struct dot_product : public operation {
		dot_product() : operation{op_type::dot_product} {}
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

	struct reduce_sum : public operation {
		reduce_sum(int dim) : operation{op_type::reduce_sum, dim} {}
	};

	struct reduce_mean : public operation {
		reduce_mean(int dim) : operation{op_type::reduce_mean, dim} {}
	};


}

