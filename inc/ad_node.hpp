#pragma once
#include <cmath>
#include <utility>
#include "tape.hpp"

struct ADNode { 
    double value; 
    int tape_idx; 
};

// Strict Baseline Functions
ADNode std_add(const ADNode& a, const ADNode& b);
ADNode std_sub(const ADNode& a, double b_val);
ADNode std_mul(const ADNode& a, const ADNode& b);
ADNode std_tanh(const ADNode& x);

// Internal Implementations
ADNode add_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval);
ADNode sub_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval);
ADNode mul_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval);
ADNode div_impl(int idx_a, double val_a, int idx_b, double val_b, bool a_rval, bool b_rval);

// Node-to-Node Operators
ADNode operator+(const ADNode& a, const ADNode& b);
ADNode operator+(ADNode&& a, const ADNode& b);
ADNode operator+(const ADNode& a, ADNode&& b);
ADNode operator+(ADNode&& a, ADNode&& b);

ADNode operator-(const ADNode& a, const ADNode& b);
ADNode operator-(ADNode&& a, const ADNode& b);
ADNode operator-(const ADNode& a, ADNode&& b);
ADNode operator-(ADNode&& a, ADNode&& b);

ADNode operator*(const ADNode& a, const ADNode& b);
ADNode operator*(ADNode&& a, const ADNode& b);
ADNode operator*(const ADNode& a, ADNode&& b);
ADNode operator*(ADNode&& a, ADNode&& b);

ADNode operator/(const ADNode& a, const ADNode& b);
ADNode operator/(ADNode&& a, const ADNode& b);
ADNode operator/(const ADNode& a, ADNode&& b);
ADNode operator/(ADNode&& a, ADNode&& b);

// Scalar Operators
#define AD_COMMUTATIVE_SCALAR(OP, DERIV_EXPR) \
    inline ADNode operator OP(const ADNode& a, double b_val) { \
        ADNode res{a.value OP b_val, global_tape.get_index()}; \
        global_tape.append_edge(res.tape_idx, a.tape_idx, (DERIV_EXPR)); \
        return res; \
    } \
    inline ADNode operator OP(ADNode&& a, double b_val) { \
        double deriv = (DERIV_EXPR); \
        if (deriv != 1.0) { \
            int head_a = global_tape.get_head(a.tape_idx); \
            global_tape.scale_async(deriv, head_a); \
        } \
        a.value = a.value OP b_val; \
        return std::move(a); \
    } \
    inline ADNode operator OP(double a_val, const ADNode& b) { return b OP a_val; } \
    inline ADNode operator OP(double a_val, ADNode&& b) { return std::move(b) OP a_val; }

#define AD_NON_COMMUTATIVE_SCALAR(OP, L_DERIV, R_VAL_EXPR, R_DERIV) \
    inline ADNode operator OP(const ADNode& a, double b_val) { \
        ADNode res{a.value OP b_val, global_tape.get_index()}; \
        global_tape.append_edge(res.tape_idx, a.tape_idx, (L_DERIV)); \
        return res; \
    } \
    inline ADNode operator OP(ADNode&& a, double b_val) { \
        double deriv = (L_DERIV); \
        if (deriv != 1.0) { \
            int head_a = global_tape.get_head(a.tape_idx); \
            global_tape.scale_async(deriv, head_a); \
        } \
        a.value = a.value OP b_val; \
        return std::move(a); \
    } \
    inline ADNode operator OP(double a_val, const ADNode& b) { \
        double val = (R_VAL_EXPR); \
        ADNode res{val, global_tape.get_index()}; \
        global_tape.append_edge(res.tape_idx, b.tape_idx, (R_DERIV)); \
        return res; \
    } \
    inline ADNode operator OP(double a_val, ADNode&& b) { \
        double val = (R_VAL_EXPR); \
        double deriv = (R_DERIV); \
        int head_b = global_tape.get_head(b.tape_idx); \
        global_tape.scale_async(deriv, head_b); \
        b.value = val; \
        return std::move(b); \
    }

AD_COMMUTATIVE_SCALAR(+, 1.0)
AD_COMMUTATIVE_SCALAR(*, b_val)
AD_NON_COMMUTATIVE_SCALAR(-, 1.0, a_val - b.value, -1.0)
AD_NON_COMMUTATIVE_SCALAR(/, 1.0 / b_val, a_val / b.value, -val / b.value)

// Transcendental Functions
ADNode tanh(const ADNode& x);
ADNode tanh(ADNode&& x);

ADNode sin(const ADNode& x);
ADNode sin(ADNode&& x);

ADNode cos(const ADNode& x);
ADNode cos(ADNode&& x);

ADNode exp(const ADNode& x);
ADNode exp(ADNode&& x);

ADNode log(const ADNode& x);
ADNode log(ADNode&& x);
