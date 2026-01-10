# Dew

Minimal expression language, multiple backends.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Overview

Dew is a minimal expression language that compiles to multiple backends. Small, ephemeral, perfectly formed—like a droplet condensed from logic. Parse once, emit to WGSL (GPU shaders), Cranelift (native JIT), or Lua (scripting).

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-dew-core` | Core types and AST |
| `rhizome-dew-scalar` | Standard scalar math functions (sin, cos, etc.) |
| `rhizome-dew-linalg` | Linear algebra types and operations (Vec2, Mat3, etc.) |

Each domain crate (scalar, linalg) includes self-contained backends:
- `wgsl` feature: WGSL shader code generation
- `lua` feature: Lua code generation + mlua execution
- `cranelift` feature: Cranelift JIT native compilation

## Architecture

```
dew-core           # Syntax only: AST, parsing
    |
    +-- dew-scalar     # Scalar domain: f32/f64 math functions
    |                  # Backends: wgsl, lua, cranelift
    |
    +-- dew-linalg     # Linalg domain: Vec2, Vec3, Mat2, Mat3, etc.
                       # Backends: wgsl, lua, cranelift
```

Crates are independent - use one or both. Each has:
- Generic over numeric type `T: Float`
- Own `FunctionRegistry<T>` and `eval<T>()`
- Self-contained backend modules

## License

MIT
