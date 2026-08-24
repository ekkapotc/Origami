#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "ad_node.hpp"
#include "tape.hpp"
#include "higher_order.hpp"

// Demonstrates genuinely automatic arbitrary-order differentiation via the
// ho_* primitives in higher_order.hpp: every derivative below is produced
// by calling ho_backward() again on the previous order's result. No
// derivative formula is hand-derived or hand-encoded in this file - unlike
// examples/higher_order_experiment.cpp, which showed the *tape itself*
// could compose hand-written derivative expressions but did not automate
// producing them.

bool check(const std::string& name, double actual, double expected, double tol = 1e-9) {
    double err = std::abs(actual - expected);
    bool passed = err < tol;
    std::cout << std::left << std::setw(30) << name
               << (passed ? "PASS" : "FAIL")
               << "  got=" << std::setprecision(10) << actual
               << "  expected=" << expected
               << "  err=" << std::scientific << std::setprecision(2) << err
               << std::defaultfloat << "\n";
    return passed;
}

int main() {
    std::thread worker(&Tape::worker_loop, &global_tape);
    bool all_passed = true;

    std::cout << "=== AUTOMATIC ARBITRARY-ORDER DIFFERENTIATION (ho_*) ===\n\n";

    // -----------------------------------------------------------------
    // f(x) = x^4, orders 1-5, each obtained by calling ho_backward() on
    // the previous order's own result node.
    // -----------------------------------------------------------------
    std::cout << "-- f(x) = x^4, repeated ho_backward() only --\n";
    {
        global_tape.clear();
        double xv = 1.5;
        ADNode x  = ho_var(xv);
        ADNode x2 = ho_mul(x, x);
        ADNode x3 = ho_mul(x2, x);
        ADNode f  = ho_mul(x3, x);

        auto g1 = ho_backward(f.tape_idx);
        all_passed &= check("order-1", g1[x.tape_idx].value, 4.0 * xv * xv * xv);

        auto g2 = ho_backward(g1[x.tape_idx].tape_idx);
        all_passed &= check("order-2", g2[x.tape_idx].value, 12.0 * xv * xv);

        auto g3 = ho_backward(g2[x.tape_idx].tape_idx);
        all_passed &= check("order-3", g3[x.tape_idx].value, 24.0 * xv);

        auto g4 = ho_backward(g3[x.tape_idx].tape_idx);
        all_passed &= check("order-4", g4[x.tape_idx].value, 24.0);

        auto g5 = ho_backward(g4[x.tape_idx].tape_idx);
        all_passed &= check("order-5 (vanishes)", g5[x.tape_idx].value, 0.0);
    }

    // -----------------------------------------------------------------
    // f(x,y) = x^2 y^3, mixed partials, purely automatic. A single
    // ho_backward() call at order k returns the whole row of order-(k+1)
    // mixed partials at once, same as the hand-derived experiment - but
    // this time the row itself was produced automatically too.
    // -----------------------------------------------------------------
    std::cout << "\n-- f(x,y) = x^2 y^3, mixed partials, repeated ho_backward() only --\n";
    {
        global_tape.clear();
        double xv = 0.8, yv = 1.3;
        ADNode x = ho_var(xv);
        ADNode y = ho_var(yv);
        ADNode f = ho_mul(ho_mul(x, x), ho_mul(ho_mul(y, y), y));

        auto g1 = ho_backward(f.tape_idx);
        all_passed &= check("df/dx", g1[x.tape_idx].value, 2.0 * xv * yv * yv * yv);
        all_passed &= check("df/dy", g1[y.tape_idx].value, 3.0 * xv * xv * yv * yv);

        auto g2 = ho_backward(g1[x.tape_idx].tape_idx);
        all_passed &= check("d2f/dx2", g2[x.tape_idx].value, 2.0 * yv * yv * yv);
        all_passed &= check("d2f/dxdy (same call)", g2[y.tape_idx].value, 6.0 * xv * yv * yv);

        auto g3 = ho_backward(g2[y.tape_idx].tape_idx);
        all_passed &= check("d3f/dxdy2", g3[y.tape_idx].value, 12.0 * xv * yv);
    }

    // -----------------------------------------------------------------
    // f(x) = tanh(x), orders 1-3, purely automatic. Orders 1-2 checked
    // against closed forms; order 3 cross-checked against a central
    // difference of the (closed-form) order-2 derivative, independent of
    // any hand-derived third-derivative algebra.
    // -----------------------------------------------------------------
    std::cout << "\n-- f(x) = tanh(x), repeated ho_backward() only --\n";
    {
        global_tape.clear();
        double xv = 0.6;
        ADNode x = ho_var(xv);
        ADNode f = ho_tanh(x);

        double t = std::tanh(xv);
        auto g1 = ho_backward(f.tape_idx);
        all_passed &= check("order-1", g1[x.tape_idx].value, 1.0 - t * t);

        auto g2 = ho_backward(g1[x.tape_idx].tape_idx);
        all_passed &= check("order-2", g2[x.tape_idx].value, -2.0 * t * (1.0 - t * t));

        auto tanh_d2 = [](double v) {
            double tv = std::tanh(v);
            return -2.0 * tv * (1.0 - tv * tv);
        };
        double h = 1e-5;
        double order3_ref = (tanh_d2(xv + h) - tanh_d2(xv - h)) / (2.0 * h);

        auto g3 = ho_backward(g2[x.tape_idx].tape_idx);
        all_passed &= check("order-3 (vs finite diff)", g3[x.tape_idx].value, order3_ref, 1e-6);
    }

    global_tape.stop();
    worker.join();

    std::cout << "\n" << (all_passed ? "All automatic higher-order checks passed." : "Some checks FAILED.") << "\n";
    return all_passed ? 0 : 1;
}
