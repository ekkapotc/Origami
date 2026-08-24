#pragma once
#include <array>
#include <atomic>
#include <cstddef>

struct ScaleTask {
    double scale;
    int snapshot_head;
};

template <size_t Capacity>
class SPSCQueue {
    std::array<ScaleTask, Capacity> ring_buffer;
    std::atomic<size_t> head{0};
    std::atomic<size_t> tail{0};

public:
    bool push(const ScaleTask& task) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Capacity;
        if (next_tail == head.load(std::memory_order_acquire)) return false; 
        
        ring_buffer[current_tail] = task;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(ScaleTask& task) {
        size_t current_head = head.load(std::memory_order_relaxed);
        if (current_head == tail.load(std::memory_order_acquire)) return false; 
        
        task = ring_buffer[current_head];
        head.store((current_head + 1) % Capacity, std::memory_order_release);
        return true;
    }
    
    bool is_empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }
};
