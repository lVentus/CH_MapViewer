Compute and rendering shaders of the LOD page pipeline.

- `lod_page_*.glsl` are included by the kernels (`#include "name"`, resolved by ComputeProgram)
  and describe shared layout only: the page pool, GPU lists, status counters, the record order
  of the birth-ordered filter, chords against the cull box and the hierarchy walk. No strategy
  keeps state in them.
- `lod_filter_*.comp` belong to the filter strategies (full scan, birth-ordered, GPU page
  culling, two-pass scatter), `lod_unfold_*.comp` to the expansion strategies (interval rules,
  DFS, BFS, persistent queue, workgroup queue).
- `lod_list_groups.comp` and `lod_draw_finish.comp` turn GPU counts into indirect dispatch and
  draw arguments; `lod_page_road.vert/.frag` draw the segment list.
