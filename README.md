# Origami

A small reverse-mode automatic differentiation (AD) engine in C++17, built
around a tape that records operations on a background thread so gradient
scaling work can overlap with graph construction on the main thread.

## Core idea: tape fusion

A standard reverse-mode AD tape allocates a new node for every intermediate
result and records one edge per input, each edge carrying a local partial
derivative:

```cpp
ADNode c = a + b; // new node C, edges C->A (1.0), C->B (1.0)
```

Chains of temporaries produced this way (`(a + b) * c`, `tanh(x * w)`, ...)
allocate a node per operation even though only the final result is ever
named. Origami's `ADNode` overloads `+`, `-`, `*`, `/`, and the transcendental
functions for `ADNode&&` (rvalue) operands. When an operand is an rvalue, the
operator does not allocate a new node — it rescales the edges already
attached to that node's head in place and reuses its index:

```cpp
ADNode c = std::move(a) + b; // no new node: A's edges are scaled, A's index is reused
```

This is what `forward()`/`step_optimized()` (vs. `forward_standard()`/
`step_standard()`) exercise in the example drivers, and it's why passing
temporaries through arithmetic (`loss = std::move(loss) + err * err`) rather
than binding them to named lvalues measurably reduces node count and tape
memory — see the `MEMORY CONSUMPTION` sections the benchmarks print.

The edge rescaling is dispatched to a background worker (`Tape::worker_loop`)
over a fixed-size SPSC ring buffer (`SPSCQueue`) instead of happening inline,
so the scale of a large fan-in subtree can be applied concurrently with the
main thread continuing to build the graph. `Tape::flush()` blocks until all
in-flight scale tasks have completed and must be called before `backward()`.

## Arbitrary-order differentiation (`ho_*`)

`backward()` returns raw `double` adjoints — a numeric dead end that can't
itself be differentiated again. `higher_order.hpp` adds a small, separate
primitive family (`ho_var`, `ho_add`, `ho_mul`, `ho_tanh`) that tags every
node it creates with its operation and operand indices in a side table.
`ho_backward()` walks that table and reconstructs each local derivative as
a genuine `ADNode` expression — built from `ho_add`/`ho_mul` themselves —
instead of a baked double. That expression is exactly as differentiable as
the original computation, so calling `ho_backward()` again on its output
differentiates one order further, with no hand-derived calculus at any
level:

```cpp
ADNode x = ho_var(1.5);
ADNode f = ho_mul(ho_mul(x, x), ho_mul(x, x)); // x^4

auto g1 = ho_backward(f.tape_idx);             // f'(x)  = 4x^3
auto g2 = ho_backward(g1[x.tape_idx].tape_idx); // f''(x) = 12x^2
auto g3 = ho_backward(g2[x.tape_idx].tape_idx); // f'''(x) = 24x
```

A single `ho_backward()` call at order *k* also returns the entire row of
order-(*k*+1) mixed partials at once (e.g. differentiating `df/dx` yields
both `d²f/dx²` and `d²f/dxdy` from one pass) — a full Hessian costs *N*
passes, not *N²*.

This is intentionally decoupled from the fast first-order path: the
rvalue-fusing operators in `ad_node.hpp` mutate an edge's weight in place
and discard which primal nodes produced it, which is incompatible with
reconstructing a symbolic, re-differentiable local derivative. `ho_*` nodes
and ordinary `ADNode` nodes share the same tape's index counter but form
two disjoint graphs — `ho_*` nodes carry no edges for `Tape::backward()`,
and ordinary nodes are invisible to `ho_backward()`. Only `+`, `*`, and
`tanh` are implemented so far. See `examples/higher_order_auto.cpp` for a
fully automatic demo (orders 1-5 on `x^4`, mixed partials on `x²y³`, orders
1-3 on `tanh(x)`) and `examples/higher_order_experiment.cpp` for the
earlier hand-derived proof of concept that motivated it.

## Layout

```
inc/          Public headers
  ad_node.hpp   ADNode + operator overloads (lvalue and rvalue forms)
  tape.hpp      Tape: node/edge storage, the async worker, backward()
  higher_order.hpp ho_* primitives + ho_backward() for arbitrary-order AD
  arena.hpp     ChunkedArena<T>: append-only, lock-free growable storage
  spsc_queue.hpp SPSCQueue<Capacity>: single-producer/single-consumer ring buffer
  dense.hpp     DenseLayer example model
  rnn.hpp       RNNLayer example model
src/          Implementations of the above
examples/     Standalone driver programs (see below); any .cpp dropped here
              is picked up automatically by the Makefile
build/, bin/  Generated object files and executables (gitignored)
```

## Building

```sh
make          # build every driver in examples/ into bin/
make run      # build, then run every driver in sequence, stopping on first failure
make tsan     # rebuild everything instrumented with ThreadSanitizer (forces a clean build)
make clean    # remove build/ and bin/
```

Each `.cpp` in `examples/` becomes its own executable at `bin/<name>`,
statically linked against the core library objects — there's no separate
test runner or framework. A driver signals failure via its process exit
code, which `make run` treats as a hard stop.

## Examples

- `ad_node_unit_test` — gradient checks: builds small expression graphs
  exercising every operator's lvalue and rvalue paths, differentiates them
  with the tape, and compares against central finite differences on an
  independent reference implementation of the same math. Exits non-zero on
  any mismatch.
- `dense_nn_benchmark`, `rnn_benchmark`, `pinn_benchmark` — larger models
  (a 3-layer dense net, an RNN doing BPTT, a physics-informed NN needing
  second derivatives) run once through `forward_standard`/`step_standard`
  and once through the fused `forward`/`step_optimized` path, reporting
  runtime, tape memory, and the max gradient discrepancy between the two
  as a correctness cross-check.
- `higher_order_auto` — the `ho_*` primitives differentiating themselves to
  arbitrary order automatically (see above), checked against closed forms
  and, for `tanh`'s third derivative, an independent finite-difference
  reference.
- `higher_order_experiment` — the earlier proof of concept: hand-derived
  derivative formulas, re-expressed as ordinary `ADNode` arithmetic on the
  original primal nodes, differentiated again via repeated `Tape::backward()`
  calls with zero changes to `Tape`/`ADNode`. Establishes that the tape
  itself composes across orders before `ho_*` automated producing the
  formulas.

## Concurrency

The tape's node/edge storage (`ChunkedArena`) and the scale-task queue
(`SPSCQueue`) are lock-free, single-producer/single-consumer structures
shared between the main thread and the one worker thread spawned per
driver via `std::thread(&Tape::worker_loop, &global_tape)`. Any change to
`Tape`, `ChunkedArena`, or `SPSCQueue` — especially to their atomic memory
orders — should be validated with `make tsan` before landing:

```sh
make tsan
for b in bin/*; do "$b" || echo "FAILED: $b"; done
```
