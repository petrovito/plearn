#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rep/ops.h>

namespace plearn::rep {


	using std::vector;
	using std::unordered_set;
	using std::unique_ptr;

	using node_id = int;
	using op_node_id = node_id;
	using tensor_node_id = node_id;

	template<typename K, typename V>
	using hash_map = std::unordered_map<K, V>;



	template<typename T>
	using read_ptr = const T*;

	template<typename T>
	using borrowed_ptr = T*;

	template<typename T>
	using owned_ptr = T*;





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
			rank{_dims.size()}, dims{_dims} {
				if (rank == 0) {
					dims.push_back(1);
					rank = 1;
				}
			}

		shape_t(std::integral auto...dims) : 
			rank{sizeof...(dims)}, dims{dims...} {}

		shape_t operator*(const shape_t& other) const {
			shape_t result;
			result.rank = rank + other.rank;
			result.dims.reserve(result.rank);
			result.dims.insert(result.dims.end(), dims.begin(), dims.end());
			result.dims.insert(result.dims.end(), other.dims.begin(), other.dims.end());
			return result;
		}

		uint64_t operator[](int i) const {
			return dims[i];
		}

		friend auto operator<=>(const shape_t&, const shape_t&) = default;
	};
	


	struct dep_type {
		bool identity_{false};
		bool independent_{false};
	};

	const dep_type identity_gradient = dep_type{true, false};
	const dep_type independent_gradient = dep_type{false, true};






	enum class run_state {
		IN_PROGRESS,
		READY,
	};

}
