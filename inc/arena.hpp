#pragma once
#include <array>
#include <atomic>
#include <cstddef>

template <typename T, size_t CHUNK_SHIFT = 16>
class ChunkedArena {
    static constexpr size_t CHUNK_SIZE = 1ULL << CHUNK_SHIFT;
    static constexpr size_t CHUNK_MASK = CHUNK_SIZE - 1;
    static constexpr size_t MAX_CHUNKS = 4096;

    std::atomic<size_t> count{0};
    std::array<std::atomic<std::array<T, CHUNK_SIZE>*>, MAX_CHUNKS> chunks{};

public:
    void push_back(const T& item) {
        size_t idx = count.load(std::memory_order_relaxed);
        size_t chunk_idx = idx >> CHUNK_SHIFT, element_idx = idx & CHUNK_MASK;
        auto* chunk = chunks[chunk_idx].load(std::memory_order_acquire);
        if (!chunk) {
            auto* fresh = new std::array<T, CHUNK_SIZE>();
            std::array<T, CHUNK_SIZE>* expected = nullptr;
            if (chunks[chunk_idx].compare_exchange_strong(expected, fresh,
                    std::memory_order_release, std::memory_order_acquire))
                chunk = fresh;
            else { delete fresh; chunk = expected; }
        }
        (*chunk)[element_idx] = item;
        count.store(idx + 1, std::memory_order_release);
    }
    
    inline T& operator[](size_t idx) {
        return (*chunks[idx >> CHUNK_SHIFT].load(std::memory_order_acquire))[idx & CHUNK_MASK];
    }

    inline const T& operator[](size_t idx) const {
        return (*chunks[idx >> CHUNK_SHIFT])[idx & CHUNK_MASK];
    }

    size_t size() const { return count.load(std::memory_order_acquire); }

    void clear() { count.store(0, std::memory_order_release); }
    
    size_t memory_bytes() const { return count.load(std::memory_order_acquire) * sizeof(T); }
};
