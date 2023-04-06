#pragma once

#include <cstdint>
#include <rep/rep_types.h>
#include <environ/env_types.h>

namespace plearn::model {
	using namespace ::plearn::rep;
	using namespace ::plearn::env;


	struct DimV {
		enum Type {
			CONST,
			VAR
		};

		DimV() = default;
		DimV(uint64_t value) : type_(CONST), value_(value) {}
		DimV(Type t) : type_(t), value_(0) {}

		Type type_;
		uint64_t value_;

		friend auto operator<=>(const DimV&, const DimV& other) = default;
	};

	const auto VAR_DIM = DimV{DimV::VAR};

	struct ShapeV {
		int rank;
		vector<DimV> dims;

		bool is_const() const {
			for (auto& dim : dims) {
				if (dim.type_ == DimV::VAR) return false;
			}
			return true;
		}

		template<typename... T>
		ShapeV(T&&... args) : rank(sizeof...(args)), dims{args...} {}

		shape_t to_shape() const {
			if (!is_const())
				throw std::runtime_error("Shape is not const");
			vector<uint64_t> dims_shape;
			for (auto& dim : dims) {
				dims_shape.push_back(dim.value_);
			}
			return {dims_shape};
		}

		friend auto operator<=>(const ShapeV&, const ShapeV& other) = default;

	};



}

