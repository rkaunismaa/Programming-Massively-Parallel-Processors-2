# Chapter 1: Introduction

## Summary
Chapter 1 introduces the motivation and context for massively parallel processing. It covers the shift from latency-oriented CPU design to throughput-oriented GPU design, the concurrency revolution, and the fundamental differences between multicore CPUs and many-thread GPUs. The chapter outlines the book's structure and teaching approach.

## Concepts Covered
- Heterogeneous parallel computing: CPU vs GPU design philosophies
- Latency-oriented vs throughput-oriented design
- The "concurrency revolution" — end of Dennard scaling since 2003
- Multicore trajectory vs many-thread trajectory
- The "peach" metaphor for application parallelizability
- Challenges in parallel programming:
  - Algorithmic complexity vs redundant work
  - Memory-bound vs compute-bound applications
  - Input data sensitivity and load imbalance
  - Synchronization overhead
- Related parallel programming interfaces: OpenMP, MPI, OpenCL
- Three overarching goals: high performance, correctness/reliability, scalability
- Book organization across four parts

## Files Generated
No compilable CUDA code in this chapter. It is purely conceptual/introductory.

## Figures Skipped (diagrams/illustrations — no compilable code)
| Figure | Description |
|--------|-------------|
| Figure 1.1 | CPU (latency-oriented) vs GPU (throughput-oriented) design comparison |
| Figure 1.2 | The "peach" metaphor — application parallelizability zones |

## Key Takeaways
- GPUs maximize throughput by having many simple cores; CPUs minimize latency with complex cores
- The performance gap between GPU and CPU peak FLOPS has been widening since 2003
- CUDA gives explicit control of parallel programming details — excellent for learning
- The book teaches parallel programming through concrete CUDA C examples
- Three goals: performance, correctness, and future hardware scalability

## Build
No build targets for this chapter.
