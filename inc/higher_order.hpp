#pragma once
#include <vector>
#include "ad_node.hpp"

// Automatic, arbitrary-order differentiation layered on top of the existing
// tape, as a self-contained primitive family: ho_var/ho_add/ho_mul/ho_tanh.
//
// Every node these primitives create is additionally tagged with its
// operation and operand indices in a side table. ho_backward() walks that
// table and reconstructs each edge's local derivative as a genuine ADNode
// expression (itself built from ho_add/ho_mul) rather than a baked double.
// That expression is exactly as differentiable as the original computation,
// so calling ho_backward() again on one of its outputs differentiates one
// order further - recursively, to any order, fully automatically (no
// hand-derived calculus required at any level).
//
// Deliberately NOT wired into the rvalue-fusing operators or the async
// scale worker in ad_node.hpp/tape.hpp: fusion mutates an edge's weight in
// place and discards which primal nodes produced it, which is incompatible
// with reconstructing a symbolic (re-differentiable) local derivative.
// ho_* nodes and ordinary ADNode nodes coexist on the same tape (both just
// consume global_tape's node index counter) but form two disjoint graphs;
// only ho_* nodes are visible to ho_backward(), and ho_* nodes carry no
// edges for the ordinary Tape::backward() to walk.
//
// Single-threaded API: build the ho_* graph and call ho_backward() only
// from the thread doing so, same as the rest of ADNode/Tape. The async
// scale worker never touches this side table, so it needs no locking.

ADNode ho_var(double value);
ADNode ho_add(const ADNode& a, const ADNode& b);
ADNode ho_mul(const ADNode& a, const ADNode& b);
ADNode ho_tanh(const ADNode& x);

// Returns one ADNode adjoint per tape node that existed when this call
// began, indexed by tape_idx (same convention as Tape::backward()). Only
// entries reachable from output_idx through ho_* nodes carry a nonzero
// value; the rest are inert zero leaves at their own index.
std::vector<ADNode> ho_backward(int output_idx);
