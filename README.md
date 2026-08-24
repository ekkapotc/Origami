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

## Layout

```
inc/          Public headers
  ad_node.hpp   ADNode + operator overloads (lvalue and rvalue forms)
  tape.hpp      Tape: node/edge storage, the async worker, backward()
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
