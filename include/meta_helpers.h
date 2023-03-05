
#include <array>
#include <cstdint>

namespace plearn {


	template<uint64_t... ints>
		struct IntSeq{
			static constexpr std::array<uint64_t, sizeof...(ints)> arr = {ints...};
		};

	template<uint64_t first, uint64_t... rest>
		constexpr uint64_t firstElem(IntSeq<first, rest...>) { return first; }

	template<uint64_t first, uint64_t... rest>
		constexpr IntSeq<rest...> allButFirst(IntSeq<first, rest...>) { return {}; }

	template<uint64_t first, uint64_t... rest>
		constexpr uint64_t lastElem(IntSeq<first, rest...>) {
			if constexpr (sizeof...(rest) == 0) return first;
			else return lastElem(IntSeq<rest...>{});
		}

	template<uint64_t... rest, uint64_t... first>
		constexpr IntSeq<first..., rest...> pack(IntSeq<first...>) { return {}; }

	template<uint64_t... first, uint64_t... last>
		constexpr auto allButLast(IntSeq<first...> firstSeq, IntSeq<last...> lastSeq) {
			const int rank = sizeof...(last);
			if constexpr (rank == 1) {
				return IntSeq<first...>{};				
			} else {
				const auto first_last = firstElem(lastSeq);
				auto rest = allButFirst(lastSeq);
				auto new_first = pack<first_last>(firstSeq);
				return allButLast(new_first, rest);
			}
		}

	template<uint64_t... last>
		constexpr auto allButLast(IntSeq<last...> lastSeq) {
			return allButLast(IntSeq<>{}, lastSeq);
		}
}
