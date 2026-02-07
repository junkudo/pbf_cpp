## Brief overview
This rule captures coding hygiene practices discovered during recent SPH kernel/test work. The goal is to prevent undefined behavior, missing dependencies, and template call-site drift.

## Explicit initialization for vectors
- **Always initialize vec2f/vec3f/Vec<Dim> explicitly before use.**
- Do **not** assume default construction zeroes memory.
- Preferred patterns:
  - `auto v = pbf::vec2f::zero();`
  - `auto v = pbf::vec3f::zero();`
  - `auto v = pbf::Vec<Dim>::zero();`
  - Value-initialization (`pbf::vec2f v{};`) is acceptable when zero() is not available.

## Template refactors must update call sites
- When templating helpers (e.g., `get_neighbors_slow`), **update all call sites** to pass explicit template args (e.g., `get_neighbors_slow<2>(...)`).
- If call sites are widely used, consider adding forwarding overloads to minimize churn.

## Headers must include what they use
- Public headers should include required standard headers directly (e.g., `<random>`), not rely on transitive includes from other files.

## Commits require explicit user request
- Do **not** create commits unless the user explicitly asks for a commit.
