#include "ad_node.hpp"

// Strict Baseline Functions
ADNode std_add(const ADNode& a, const ADNode& b) {
    ADNode res{a.value + b.value, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, a.tape_idx, 1.0);
    global_tape.append_edge(res.tape_idx, b.tape_idx, 1.0);
    return res;
}

ADNode std_sub(const ADNode& a, double b_val) {
    ADNode res{a.value - b_val, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, a.tape_idx, 1.0);
    return res;
}

ADNode std_mul(const ADNode& a, const ADNode& b) {
    ADNode res{a.value * b.value, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, a.tape_idx, b.value);
    global_tape.append_edge(res.tape_idx, b.tape_idx, a.value);
    return res;
}

ADNode std_tanh(const ADNode& x) {
    double th = std::tanh(x.value);
    ADNode res{th, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, 1.0 - th * th);
    return res;
}

// Internal Node Implementations
ADNode add_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval) {
    if (a_rval && idx_a > idx_b) {
        global_tape.append_edge(idx_a, idx_b, 1.0);
        return {val_a + val_b, idx_a};
    } else if (b_rval && idx_b > idx_a) {
        global_tape.append_edge(idx_b, idx_a, 1.0);
        return {val_a + val_b, idx_b};
    } else {
        ADNode res{val_a + val_b, global_tape.get_index()};
        global_tape.append_edge(res.tape_idx, idx_a, 1.0);
        global_tape.append_edge(res.tape_idx, idx_b, 1.0);
        return res;
    }
}

ADNode sub_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval) {
    if (a_rval && idx_a > idx_b) {
        global_tape.append_edge(idx_a, idx_b, -1.0);
        return {val_a - val_b, idx_a};
    } else if (b_rval && idx_b > idx_a) {
        int head_b = global_tape.get_head(idx_b);
        global_tape.scale_async(-1.0, head_b);
        global_tape.append_edge(idx_b, idx_a, 1.0);
        return {val_a - val_b, idx_b};
    } else {
        ADNode res{val_a - val_b, global_tape.get_index()};
        global_tape.append_edge(res.tape_idx, idx_a, 1.0);
        global_tape.append_edge(res.tape_idx, idx_b, -1.0);
        return res;
    }
}

ADNode mul_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval) {
    if (a_rval && idx_a > idx_b) {
        int head_a = global_tape.get_head(idx_a);
        global_tape.scale_async(val_b, head_a);      
        global_tape.append_edge(idx_a, idx_b, val_a); 
        return {val_a * val_b, idx_a};
    } else if (b_rval && idx_b > idx_a) {
        int head_b = global_tape.get_head(idx_b);
        global_tape.scale_async(val_a, head_b);      
        global_tape.append_edge(idx_b, idx_a, val_b); 
        return {val_a * val_b, idx_b};
    } else {
        ADNode res{val_a * val_b, global_tape.get_index()};
        global_tape.append_edge(res.tape_idx, idx_a, val_b);
        global_tape.append_edge(res.tape_idx, idx_b, val_a);
        return res;
    }
}

ADNode div_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval) {
    double d_a = 1.0 / val_b;
    double d_b = -val_a / (val_b * val_b);
    if (a_rval && idx_a > idx_b) {
        int head_a = global_tape.get_head(idx_a);
        global_tape.scale_async(d_a, head_a);
        global_tape.append_edge(idx_a, idx_b, d_b);
        return {val_a / val_b, idx_a};
    } else if (b_rval && idx_b > idx_a) {
        int head_b = global_tape.get_head(idx_b);
        global_tape.scale_async(d_b, head_b);
        global_tape.append_edge(idx_b, idx_a, d_a);
        return {val_a / val_b, idx_b};
    } else {
        ADNode res{val_a / val_b, global_tape.get_index()};
        global_tape.append_edge(res.tape_idx, idx_a, d_a);
        global_tape.append_edge(res.tape_idx, idx_b, d_b);
        return res;
    }
}

// Node-to-Node Operators
ADNode operator+(const ADNode& a, const ADNode& b) { return add_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, false); }
ADNode operator+(ADNode&& a, const ADNode& b) { return add_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, false); }
ADNode operator+(const ADNode& a, ADNode&& b) { return add_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, true); }
ADNode operator+(ADNode&& a, ADNode&& b) { return add_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, true); }

ADNode operator-(const ADNode& a, const ADNode& b) { return sub_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, false); }
ADNode operator-(ADNode&& a, const ADNode& b) { return sub_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, false); }
ADNode operator-(const ADNode& a, ADNode&& b) { return sub_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, true); }
ADNode operator-(ADNode&& a, ADNode&& b) { return sub_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, true); }

ADNode operator*(const ADNode& a, const ADNode& b) { return mul_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, false); }
ADNode operator*(ADNode&& a, const ADNode& b) { return mul_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, false); }
ADNode operator*(const ADNode& a, ADNode&& b) { return mul_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, true); }
ADNode operator*(ADNode&& a, ADNode&& b) { return mul_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, true); }

ADNode operator/(const ADNode& a, const ADNode& b) { return div_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, false); }
ADNode operator/(ADNode&& a, const ADNode& b) { return div_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, false); }
ADNode operator/(const ADNode& a, ADNode&& b) { return div_impl(a.tape_idx, a.value, b.tape_idx, b.value, false, true); }
ADNode operator/(ADNode&& a, ADNode&& b) { return div_impl(a.tape_idx, a.value, b.tape_idx, b.value, true, true); }

// Transcendental Functions
ADNode tanh(const ADNode& x) {
    double th = std::tanh(x.value);
    ADNode res{th, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, 1.0 - th * th);
    return res;
}
ADNode tanh(ADNode&& x) {
    double th = std::tanh(x.value);
    int head_x = global_tape.get_head(x.tape_idx);
    global_tape.scale_async(1.0 - th * th, head_x);
    return {th, x.tape_idx};
}

ADNode sin(const ADNode& x) {
    double s = std::sin(x.value);
    ADNode res{s, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, std::cos(x.value));
    return res;
}
ADNode sin(ADNode&& x) {
    double s = std::sin(x.value);
    int head_x = global_tape.get_head(x.tape_idx);
    global_tape.scale_async(std::cos(x.value), head_x);
    return {s, x.tape_idx};
}

ADNode cos(const ADNode& x) {
    double c = std::cos(x.value);
    ADNode res{c, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, -std::sin(x.value));
    return res;
}
ADNode cos(ADNode&& x) {
    double c = std::cos(x.value);
    int head_x = global_tape.get_head(x.tape_idx);
    global_tape.scale_async(-std::sin(x.value), head_x);
    return {c, x.tape_idx};
}

ADNode exp(const ADNode& x) {
    double e = std::exp(x.value);
    ADNode res{e, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, e);
    return res;
}
ADNode exp(ADNode&& x) {
    double e = std::exp(x.value);
    int head_x = global_tape.get_head(x.tape_idx);
    global_tape.scale_async(e, head_x);
    return {e, x.tape_idx};
}

ADNode log(const ADNode& x) {
    double l = std::log(x.value);
    ADNode res{l, global_tape.get_index()};
    global_tape.append_edge(res.tape_idx, x.tape_idx, 1.0 / x.value);
    return res;
}
ADNode log(ADNode&& x) {
    double l = std::log(x.value);
    int head_x = global_tape.get_head(x.tape_idx);
    global_tape.scale_async(1.0 / x.value, head_x);
    return {l, x.tape_idx};
}
