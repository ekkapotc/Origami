#include "higher_order.hpp"
#include "tape.hpp"
#include <cmath>
#include <unordered_map>

namespace {

enum class DiffOp { LEAF, ADD, MUL, TANH };

struct DiffOpRecord {
    double value;
    DiffOp op;
    int lhs;
    int rhs; // -1 when unused (LEAF, TANH's unary operand slot)
};

std::unordered_map<int, DiffOpRecord>& registry() {
    static std::unordered_map<int, DiffOpRecord> reg;
    return reg;
}

ADNode make_node(double value, DiffOp op, int lhs, int rhs) {
    int idx = global_tape.get_index();
    registry()[idx] = {value, op, lhs, rhs};
    return {value, idx};
}

void accumulate(std::vector<ADNode>& adjoints, std::vector<char>& has, int idx, const ADNode& contribution) {
    if (idx < 0) return;
    if (!has[idx]) {
        adjoints[idx] = contribution;
        has[idx] = 1;
    } else {
        adjoints[idx] = ho_add(adjoints[idx], contribution);
    }
}

} // namespace

ADNode ho_var(double value) {
    return make_node(value, DiffOp::LEAF, -1, -1);
}

ADNode ho_add(const ADNode& a, const ADNode& b) {
    return make_node(a.value + b.value, DiffOp::ADD, a.tape_idx, b.tape_idx);
}

ADNode ho_mul(const ADNode& a, const ADNode& b) {
    return make_node(a.value * b.value, DiffOp::MUL, a.tape_idx, b.tape_idx);
}

ADNode ho_tanh(const ADNode& x) {
    return make_node(std::tanh(x.value), DiffOp::TANH, x.tape_idx, -1);
}

std::vector<ADNode> ho_backward(int output_idx) {
    auto& reg = registry();
    int total = static_cast<int>(global_tape.get_node_count());

    std::vector<ADNode> adjoints;
    adjoints.reserve(total);
    for (int i = 0; i < total; ++i) adjoints.push_back({0.0, i});
    std::vector<char> has(total, 0);

    adjoints[output_idx] = ho_var(1.0);
    has[output_idx] = 1;

    for (int i = total - 1; i >= 0; --i) {
        if (!has[i]) continue;
        auto it = reg.find(i);
        if (it == reg.end()) continue; // not an ho_* node: a terminal w.r.t. this graph

        const DiffOpRecord& rec = it->second;
        ADNode upstream = adjoints[i];

        switch (rec.op) {
            case DiffOp::LEAF:
                break;
            case DiffOp::ADD:
                accumulate(adjoints, has, rec.lhs, upstream);
                accumulate(adjoints, has, rec.rhs, upstream);
                break;
            case DiffOp::MUL: {
                ADNode a_node{reg.at(rec.lhs).value, rec.lhs};
                ADNode b_node{reg.at(rec.rhs).value, rec.rhs};
                accumulate(adjoints, has, rec.lhs, ho_mul(upstream, b_node));
                accumulate(adjoints, has, rec.rhs, ho_mul(upstream, a_node));
                break;
            }
            case DiffOp::TANH: {
                // d/dx tanh(x) = 1 - tanh(x)^2, expressed via ho_* so the
                // formula itself stays differentiable for the next order.
                ADNode th_node{rec.value, i};
                ADNode one = ho_var(1.0);
                ADNode neg_one = ho_var(-1.0);
                ADNode th_sq = ho_mul(th_node, th_node);
                ADNode deriv = ho_add(one, ho_mul(neg_one, th_sq));
                accumulate(adjoints, has, rec.lhs, ho_mul(upstream, deriv));
                break;
            }
        }
    }

    return adjoints;
}
