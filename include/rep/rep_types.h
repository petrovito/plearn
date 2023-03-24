#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace plearn::rep {


	using std::vector;
	using std::unordered_set;
	using std::unique_ptr;

	using node_id = int;
	using op_node_id = node_id;
	using tensor_node_id = node_id;

	template<typename K, typename V>
	using hash_map = std::unordered_map<K, V>;




	struct shape_t {
		int rank;
		vector<uint64_t> dims;

		uint64_t size() const {
			std::size_t size = 1;
			for (auto dim : dims) {
				size *= dim;
			}
			return size;
		}
		shape_t() = default;

		/* template<typename IntType, template <typename> class coll> */
		/* shape(const coll<IntType>& _dims) : */ 
		/* 	rank{_dims.size()}, dims{} { */
		/* 		std::copy(_dims.begin(), _dims.end(), dims.begin()); */
		/* 	} */

		shape_t(vector<uint64_t> _dims) : 
			rank{_dims.size()}, dims{_dims} {}

		shape_t(std::integral auto...dims) : 
			rank{sizeof...(dims)}, dims{dims...} {}

		friend auto operator<=>(const shape_t&, const shape_t&) = default;
	};
	










	enum class op_type {
		noop,
		identity,
		matmul,
		vecmatmul,
		add,
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


	enum class run_state {
		IN_PROGRESS,
		READY,
	};

}
