# Chapter 15: Graph Traversal

Breadth-first search (BFS) implementations on a 9-vertex, 15-edge directed graph (Fig 15.1). Five progressively optimized kernels from the book plus two exercises.

## Kernels

| Binary | Source | Description | Data Structure |
|--------|--------|-------------|---------------|
| `ch15_bfs_push` | `ch15_bfs_push.cu` | Fig 15.6 — Vertex-centric push (top-down), one thread per vertex | CSR |
| `ch15_bfs_pull` | `ch15_bfs_pull.cu` | Fig 15.8 — Vertex-centric pull (bottom-up), one thread per vertex | CSC |
| `ch15_bfs_edge` | `ch15_bfs_edge.cu` | Fig 15.10 — Edge-centric, one thread per edge | COO |
| `ch15_bfs_frontier` | `ch15_bfs_frontier.cu` | Fig 15.12 — Push with frontiers, atomicCAS for label+insert | CSR |
| `ch15_bfs_privatized` | `ch15_bfs_privatized.cu` | Fig 15.14 — Push with privatized (shared-mem) frontiers per block | CSR |
| `ch15_bfs_direction_opt` | `ch15_bfs_direction_opt.cu` | Exercise 2 — Direction-optimized: push→pull switch | CSR + CSC |
| `ch15_bfs_singleblock` | `ch15_bfs_singleblock.cu` | Exercise 3 — Single-block multi-level kernel for small frontiers | CSR |

## Graph: 9 vertices, 15 directional edges (Fig 15.1)

```
0 → 1, 2
1 → 3, 4
2 → 5, 6, 7
3 → 4, 8
4 → 5, 8
5 → 8
6 → 8
7 → 0
8 → 1
```

BFS from root 0 yields levels: `0 1 1 2 2 2 2 2 3`

## Validation Results (all PASS)

| Kernel | Validation | BFS Levels (vertices 0-8) |
|--------|:----------:|---------------------------|
| Push (Fig 15.6) | PASS | 0 1 1 2 2 2 2 2 3 |
| Pull (Fig 15.8) | PASS | 0 1 1 2 2 2 2 2 3 |
| Edge-centric (Fig 15.10) | PASS | 0 1 1 2 2 2 2 2 3 |
| Frontier (Fig 15.12) | PASS | 0 1 1 2 2 2 2 2 3 |
| Privatized (Fig 15.14) | PASS | 0 1 1 2 2 2 2 2 3 |
| Direction-opt (Ex2) | PASS | 0 1 1 2 2 2 2 2 3 |
| Single-block (Ex3) | PASS | 0 1 1 2 2 2 2 2 3 |

## Implementation Comparison

| Approach | Thread Assignment | Data | Atomics | Frontier | Key Feature |
|----------|:-----------------:|:----:|:-------:|:--------:|-------------|
| Push | 1 per vertex | CSR | None | Implicit | Simple, only prev-level threads work |
| Pull | 1 per vertex | CSC | None | Implicit | Early break per vertex |
| Edge-Centric | 1 per edge | COO | None | Implicit | Uniform work, more parallelism |
| Frontier | 1 per prev-frontier element | CSR | atomicCAS + atomicAdd | Explicit arrays | No wasted threads |
| Privatized | 1 per prev-frontier element | CSR | atomicCAS + shared-mem atomics | Shared + global | Low contention via privatization |
| Direction-Opt | 1 per vertex (varies) | CSR + CSC | None | Implicit | Push for early levels, pull for later |
| Single-Block | 1 block | CSR | atomicCAS + shared-mem atomics | Shared only | Multiple levels in one launch |

## Key Observations

1. **Push vs Pull**: Push iterates over neighbors of vertices in the previous level (fewer threads work, but each loop is full). Pull has every unvisited thread search for a neighbor in the previous level (more parallelism, early break possible). Push better for early levels (small frontier, many unvisited), pull better for later levels (large frontier, few unvisited).

2. **Edge-centric** exposes more parallelism (threads = edges ≥ vertices) and uniform work per thread, at the cost of checking every edge in every level.

3. **Frontiers** eliminate wasted threads — only previous-level vertices are processed. atomicCAS prevents double-insertion into the frontier.

4. **Privatization** reduces contention on the frontier counter by using per-block shared-memory counters, then coalesced-writing to global memory.

5. **Direction optimization** switches from push (CSR) to pull (CSC) when `frontier_size × α > unvisited_count`. Our switch triggered at level 2 (frontier=5 > unvisited=1).

6. **Single-block kernel** processes multiple levels without grid launch overhead when frontiers are small. The entire 9-vertex BFS (3 levels) fits in one block launch.

## Storage Formats

| Format | Storage | Access Pattern | Use Case |
|--------|:-------:|----------------|----------|
| CSR | srcPtrs + dst | Outgoing edges of a vertex | Push (top-down) |
| CSC | dstPtrs + src | Incoming edges to a vertex | Pull (bottom-up) |
| COO | src + dst | Source and destination of an edge | Edge-centric |

## Building & Running

```bash
nvcc -std=c++17 -arch=sm_61 -O2 -o ch15_bfs_push ch15_bfs_push.cu && ./ch15_bfs_push
```

All binaries target the GTX 1050 (sm_61, device 1). Timings are dominated by kernel launch overhead for the tiny 9-vertex graph; meaningful performance comparisons require larger graphs.

## Files

```
ch15_graph_traversal/
├── ch15_bfs_push.cu              # Fig 15.6 — Vertex-centric push
├── ch15_bfs_pull.cu              # Fig 15.8 — Vertex-centric pull
├── ch15_bfs_edge.cu              # Fig 15.10 — Edge-centric
├── ch15_bfs_frontier.cu          # Fig 15.12 — Push with frontiers
├── ch15_bfs_privatized.cu        # Fig 15.14 — Privatized frontiers
├── ch15_bfs_direction_opt.cu     # Exercise 2 — Direction-optimized
├── ch15_bfs_singleblock.cu       # Exercise 3 — Single-block multi-level
├── README.md                     # This file
└── (binaries)                    # 7 compiled executables
```
