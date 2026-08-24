#include "tape.hpp"

Tape global_tape;

void Tape::set_wait_strategy(WaitStrategy strategy) { 
    worker_strategy = strategy; 
}

int Tape::get_index() {
    node_heads.push_back(-1);
    return node_heads.size() - 1;
}

int Tape::get_head(int idx) { 
    return node_heads[idx]; 
}

void Tape::append_edge(int p_idx, int c_idx, double w) {
    edges.push_back({c_idx, w, node_heads[p_idx]});
    node_heads[p_idx] = edges.size() - 1;
}

void Tape::scale_async(double scale, int snapshot_head) {
    if (scale == 1.0) return; 
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    
    while (!task_queue.push({scale, snapshot_head})) {
        std::this_thread::yield(); 
    }
}

void Tape::worker_loop() {
    ScaleTask task;
    int idle_spins = 0;

    while (active.load(std::memory_order_acquire) || tasks_in_flight.load(std::memory_order_relaxed) > 0) {
        if (task_queue.pop(task)) {
            idle_spins = 0; 
            int curr = task.snapshot_head;
            while (curr != -1) { 
                edges[curr].weight *= task.scale; 
                curr = edges[curr].next_edge_idx; 
            }
            tasks_in_flight.fetch_sub(1, std::memory_order_release);
        } else {
            if (worker_strategy == WaitStrategy::YIELD) {
                std::this_thread::yield();
            } else {
                apply_exponential_backoff(idle_spins);
            }
        }
    }
}

void Tape::flush() {
    while (tasks_in_flight.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
    }
}

void Tape::clear() {
    flush();
    edges.clear();
    node_heads.clear();
}

void Tape::stop() { 
    active.store(false, std::memory_order_release); 
}

std::vector<double> Tape::backward(int output_idx) {
    std::vector<double> adjoints(node_heads.size(), 0.0);
    adjoints[output_idx] = 1.0;
    
    for (int i = node_heads.size() - 1; i >= 0; --i) {
        int curr = node_heads[i];
        while (curr != -1) {
            adjoints[edges[curr].child_idx] += edges[curr].weight * adjoints[i];
            curr = edges[curr].next_edge_idx;
        }
    }
    return adjoints;
}

size_t Tape::get_node_count() const { return node_heads.size(); }
size_t Tape::get_edge_count() const { return edges.size(); }
size_t Tape::get_memory_bytes() const {
    return node_heads.memory_bytes() + edges.memory_bytes();
}
