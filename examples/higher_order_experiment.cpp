#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "ad_node.hpp"
#include "tape.hpp"

// Feasibility experiment for arbitrarily-high-order differentiation using
// the *existing* Tape/ADNode implementation, unmodified.
//
// Tape::backward(output_idx) is a pure function of the accumulated tape
// state: it seeds adjoints[output_idx] = 1 and walks every edge ever
// recorded. Nothing stops us from calling it on a node built *from* the
// primal inputs of an earlier computation, rather than on the original
// loss node. So if a human (or a symbolic-diff pass) writes the derivative
// of some expression as a new ADNode expression referencing the same
// primal ADNodes, that derivative expression is itself just as
// differentiable as the original — and backward() can be called on it
// again, and again, composing to arbitrary order.
//
// What this experiment does NOT show: automatic construction of that
// derivative expression. Every "order k+1" formula below is hand-derived
// calculus, re-expressed in ADNode arithmetic. Tape::backward() itself
// still returns raw doubles (a numeric dead end, not a new tape node) —
// see the README-worthy writeup this program's output feeds into for what
// full automation would require.

bool check(const std::string& name, double actual, double expected, double tol = 1e-9) {
    double err = std::abs(actual - expected);
    bool passed = err < tol;
    std::cout << std::left << std::setw(46) << name
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

    std::cout << "=== HIGHER-ORDER DIFFERENTIATION FEASIBILITY EXPERIMENT ===\n\n";

    // -----------------------------------------------------------------
    // Experiment 1: f(x) = x^4, orders 1..4, via nested backward() calls
    // on hand-derived (but tape-connected) derivative expressions.
    // -----------------------------------------------------------------
    std::cout << "-- f(x) = x^4, single variable, orders 1-4 --\n";
    {
        global_tape.clear();
        double xv = 1.5;
        ADNode x{xv, global_tape.get_index()};
        ADNode y = x * x * x * x;

        global_tape.flush();
        auto g1 = global_tape.backward(y.tape_idx);
        all_passed &= check("order-1 (automatic, via backward)", g1[x.tape_idx], 4.0 * xv * xv * xv);

        ADNode d1 = x * x * x * 4.0;   // hand-derived f'(x) = 4x^3, built on the SAME x
        global_tape.flush();
        auto g2 = global_tape.backward(d1.tape_idx);
        all_passed &= check("order-2 (backward on hand-derived d1)", g2[x.tape_idx], 12.0 * xv * xv);

        ADNode d2 = x * x * 12.0;      // f''(x) = 12x^2
        global_tape.flush();
        auto g3 = global_tape.backward(d2.tape_idx);
        all_passed &= check("order-3 (backward on hand-derived d2)", g3[x.tape_idx], 24.0 * xv);

        ADNode d3 = x * 24.0;          // f'''(x) = 24x
        global_tape.flush();
        auto g4 = global_tape.backward(d3.tape_idx);
        all_passed &= check("order-4 (backward on hand-derived d3)", g4[x.tape_idx], 24.0);
    }

    // -----------------------------------------------------------------
    // Experiment 2: f(x,y) = x^2 * y^3. Demonstrates that ONE backward()
    // call at order k returns the entire row of mixed partials at order
    // k+1 simultaneously (a full Hessian row per pass, not per entry).
    // -----------------------------------------------------------------
    std::cout << "\n-- f(x,y) = x^2 y^3, two variables, mixed partials --\n";
    {
        global_tape.clear();
        double xv = 0.8, yv = 1.3;
        ADNode x{xv, global_tape.get_index()};
        ADNode y{yv, global_tape.get_index()};
        ADNode f = x * x * y * y * y;

        global_tape.flush();
        auto g1 = global_tape.backward(f.tape_idx);
        all_passed &= check("df/dx", g1[x.tape_idx], 2.0 * xv * yv * yv * yv);
        all_passed &= check("df/dy", g1[y.tape_idx], 3.0 * xv * xv * yv * yv);

        ADNode dfdx = x * 2.0 * y * y * y;   // df/dx = 2x y^3
        global_tape.flush();
        auto g2 = global_tape.backward(dfdx.tape_idx);
        all_passed &= check("d2f/dx2  (from backward on dfdx)", g2[x.tape_idx], 2.0 * yv * yv * yv);
        all_passed &= check("d2f/dxdy (SAME backward call, for free)", g2[y.tape_idx], 6.0 * xv * yv * yv);

        ADNode d2fdxdy = x * 6.0 * y * y;    // d2f/dxdy = 6x y^2
        global_tape.flush();
        auto g3 = global_tape.backward(d2fdxdy.tape_idx);
        all_passed &= check("d3f/dxdy2 (third order, mixed)", g3[y.tape_idx], 12.0 * xv * yv);
    }

    // -----------------------------------------------------------------
    // Experiment 3: f(x) = sin(x). Derivatives cycle sin -> cos -> -sin
    // -> -cos, exercising the transcendental primitives to order 3.
    // -----------------------------------------------------------------
    std::cout << "\n-- f(x) = sin(x), transcendental, orders 1-3 --\n";
    {
        global_tape.clear();
        double xv = 0.6;
        ADNode x{xv, global_tape.get_index()};
        ADNode s = sin(x);

        global_tape.flush();
        auto g1 = global_tape.backward(s.tape_idx);
        all_passed &= check("order-1 (automatic, via backward)", g1[x.tape_idx], std::cos(xv));

        ADNode d1 = cos(x);              // f'(x) = cos(x)
        global_tape.flush();
        auto g2 = global_tape.backward(d1.tape_idx);
        all_passed &= check("order-2 (backward on hand-derived d1)", g2[x.tape_idx], -std::sin(xv));

        ADNode d2 = sin(x) * (-1.0);     // f''(x) = -sin(x)
        global_tape.flush();
        auto g3 = global_tape.backward(d2.tape_idx);
        all_passed &= check("order-3 (backward on hand-derived d2)", g3[x.tape_idx], -std::cos(xv));
    }

    global_tape.stop();
    worker.join();

    std::cout << "\n" << (all_passed ? "All higher-order checks passed." : "Some higher-order checks FAILED.") << "\n";
    return all_passed ? 0 : 1;
}
