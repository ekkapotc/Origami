#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include "spsc_queue.hpp"
#include "arena.hpp"

struct TapeEdge { 
    int child_idx; 
    double weight; 
    int next_edge_idx; 
};

enum class WaitStrategy { YIELD, EXPONENTIAL_BACKOFF };

class Tape {
    ChunkedArena<TapeEdge> edges;
    ChunkedArena<int> node_heads;
    
    SPSCQueue<65536> task_queue; 
    std::atomic<bool> active{true};
    std::atomic<int> tasks_in_flight{0}; 
    
    WaitStrategy worker_strategy = WaitStrategy::EXPONENTIAL_BACKOFF;

public:
    void set_wait_strategy(WaitStrategy strategy);
    int get_index();
    int get_head(int idx);
    void append_edge(int p_idx, int c_idx, double w);
    void scale_async(double scale, int snapshot_head);
    void worker_loop();
    void flush();
    void clear();
    void stop();
    std::vector<double> backward(int output_idx);
    size_t get_node_count() const;
    size_t get_edge_count() const;
    size_t get_memory_bytes() const;

private:
    inline void apply_exponential_backoff(int& idle_spins) {
        idle_spins++;
        if (idle_spins < 16) {
        } else if (idle_spins < 64) {
            std::this_thread::yield();
        } else {
            int sleep_us = std::min(idle_spins - 64, 1000);
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }
};

extern Tape global_tape;
