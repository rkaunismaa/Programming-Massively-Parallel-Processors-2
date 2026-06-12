# Chapter 19 — Parallel Programming and Computational Thinking

**Type**: Conceptual (no CUDA code)
**Pages**: 442-444 (3 pages)

## Overview

This chapter shifts from practical CUDA implementation to abstract concepts
of parallel programming design. It presents computational thinking as a framework
for algorithm selection, problem decomposition, and tradeoff analysis.

## 19.1 Goals of Parallel Computing

Three primary motivations for parallel computing:

1. **Solve problems in less time** — Meet tighter deadlines (e.g., overnight risk
   analysis that must complete before market open).

2. **Solve bigger problems in the same time** — Handle growing datasets without
   exceeding time budgets (e.g., expanding portfolio size).

3. **Achieve better solutions** — Use more accurate models that would be
   computationally infeasible on sequential hardware (e.g., considering more
   risk factor interactions).

In practice, parallel computing is often driven by combinations of these goals.
The common thread: **increased speed** enables all three.

## 19.2 Algorithm Selection

Key considerations when selecting parallel algorithms:

| Aspect | Tradeoff |
|--------|----------|
| **Computational steps** | Fewer steps vs more parallelism |
| **Degree of parallelism** | More parallelism vs work efficiency |
| **Numerical stability** | Accuracy vs performance |
| **Memory bandwidth** | Bandwidth usage vs compute intensity |

There is rarely a single algorithm that wins on all four dimensions. For a given
hardware system, the programmer must select the best compromise.

**Example from Ch18**: Direct Coulomb Summation (accurate, O(n²)) vs cutoff
binning (approximate, O(n)). The choice depends on required accuracy and system size.

## 19.3 Problem Decomposition

The process of breaking a domain problem into coordinated work units:

1. **Identify modules** with different computational requirements
2. **Decide what runs on GPU vs CPU** based on:
   - Amount of work per module (small work → not worth GPU overhead)
   - Data dependencies between modules
   - Overlap potential (can GPU and CPU work run concurrently?)

**Molecular dynamics example (Fig 19.1)**:
- Nonbonded forces: 95% of time, massive parallelism → GPU
- Vibrational/rotational forces: small computation → CPU
- Position/velocity update: serial dependency → CPU

**Amdahl's Law**: Speedup is limited by the sequential portion.
If 95% is accelerated 100×, overall speedup = 1/(5% + 95%/100) ≈ 17×.
If GPU/CPU execution overlaps, speedup = 1/(5%) = 20×.

## 19.4 Computational Thinking

Core principles for parallel programming design:

1. **Analyze the problem structure** — Identify inherent parallelism vs serial
   dependencies. Which parts are independent? Which require ordering?

2. **Transform the problem when beneficial** — Loop interchange (Ch17, Ch18),
   scatter→gather conversion, data layout reorganization. Good computational
   thinkers restructure problems, not just map them directly.

3. **Balance parallelism, work efficiency, and resource consumption** — More
   parallelism often means more total work (redundant computation). The art is
   finding the sweet spot for the target hardware.

4. **Domain knowledge matters** — Understanding the physical/domain meaning
   of approximate solutions (e.g., cutoff binning accuracy vs clinical acceptability
   in MRI) enables aggressive optimizations that maintain practical utility.

## Key Takeaways

- Parallel computing is about **speed**, which enables all three goals
- **Algorithm selection** is always a multi-dimensional tradeoff
- **Problem decomposition** decides GPU/CPU split — Amdahl's Law governs the ceiling
- **Computational thinking** includes restructuring problems, not just mapping them
- **Domain expertise** unlocks optimizations that pure CS knowledge cannot

## Relationship to PMPP Chapters

| Chapter | Computational Thinking Example |
|---------|-------------------------------|
| 6 | Thread coarsening: more work/thread, fewer threads |
| 7 | Tiling: redundant halo loads trade compute for bandwidth |
| 8 | Register tiling: eliminating shared memory for z-neighbors |
| 12 | Co-rank: dynamic input identification enables parallel merge |
| 17 | Loop interchange: scatter→gather, hardware trig tradeoffs |
| 18 | Algorithm selection: DCS vs cutoff binning, O(n²) vs O(n) |
