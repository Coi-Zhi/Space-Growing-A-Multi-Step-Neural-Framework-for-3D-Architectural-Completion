# MSc-Thesis-Arch-RAG-Expanding
utilizing voxel grids as a fundamental spatial syntax,Retrieval-Augmented Generation (RAG) system driven by a Transformer architecture. The model generates structures iteratively through two distinct stages: a "Bridge" phase,and a "Grow" phase.

<p align="center">
  <img src="Medias/Gif/02_0.gif" width="48%" />
  <img src="Medias/Gif/02.gif" width="48%" />
</p>
<p align="center">
  <b>Showcase 1</b>
</p>
<p align="center">
  <img src="Medias/Gif/04_0.gif" width="48%" />
  <img src="Medias/Gif/04.gif" width="48%" />
</p>
<p align="center">
  <b>Showcase 2</b>
</p>

---

## From Isovist to Space

<p align="center">
  <img src="Medias/Gif/Isovist.gif" width="50%" />
  <br>
  <em>Isovist Sample</em>
</p>

An isovist is an egocentric spatial descriptor that defines the volume of space visible or directly accessible from a single vantage point along line-of-sight rays. While fundamentally three-dimensional, isovists are frequently analyzed via two-dimensional planar or vertical cross-sections. Every point in physical space corresponds to a unique isovist polygon or polyhedron, making it a foundational representation for spatial analysis and visibility structure.

---

## Dataset Preparation Pipeline

Synthetic spatial data is generated in **Unreal Engine 5** via a simulated first-person capture workflow:

1. **Trajectory Recording:** First-person roaming paths are defined across 3D architectural environments, logging continuous 6-DoF camera poses.
2. **Multi-Pass Capture:** Synchronous RGB frames and scene depth buffers are exported alongside camera intrinsics.
3. **Voxel Discretization:** 2.5D depth maps are back-projected into 3D metric point clouds and discretized into uniform voxel grids.

---
## Experiments & Architectural Exploration

To identify the optimal spatial representation for long-sequence architectural completion, several representations and dataflows were systematically evaluated:

* **1. Direct Point-Level Autoregression (Point-by-Point Prediction)**
  * **Approach:** Directly regress continuous $(x, y, z)$ coordinates using raw point sequence inputs.
  * **Bottlenecks:** Incurs quadratic computational complexity $O(N^2)$ as point density scales; leads to severe cumulative spatial drift and topological collapse during multi-step rollout.

* **2. Modular / Chunk-Based Assembly ("Lego" Representation)**
  * **Approach:** Discretize 3D spaces into pre-segmented geometric chunks and predict spatial assembly.
  * **Bottlenecks:** Introduces arbitrary boundary segmentation errors; rigid modular constraints struggle to generalize across diverse non-standard architectural typologies.

* **3. Sub-Space & Latent Coordinate Tokenization (Quantized Coordinate Mapping)**
  * **Approach:** Map Euclidean space into high-dimensional latent tokens or coordinate indices to discretize the generative field.
  * **Bottlenecks:** Unconstrained coordinate spaces cause combinatorial explosion and GPU memory saturation (OOM); requires careful sparse bounds to balance resolution against computation.

---

### Key Takeaway
These exploratory iterations led to the **Space Growing framework**: adopting a **sparse hierarchical voxel representation with directional autoregression**, achieving the optimal Pareto frontier between geometric expressiveness, boundary coherence, and hardware computational budget.

---

Path File Download From Here []

