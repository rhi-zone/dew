# Introduction

Dew is an expression language for procedural generation. It provides composable math expressions that compile to multiple backends.

Part of the [Rhizome](https://rhizome-lab.github.io) ecosystem.

## Architecture

```
dew-core               # Syntax only: AST, parsing
    |
    +-- dew-cond       # Conditional backend helpers
    |
    +-- dew-scalar     # Scalar domain: f32/f64 math functions
    |
    +-- dew-linalg     # Linalg domain: Vec2, Vec3, Mat2, Mat3
    |
    +-- dew-complex    # Complex numbers: [re, im]
    |
    +-- dew-quaternion # Quaternions: [x, y, z, w], Vec3
```

All domain crates have WGSL, Lua, and Cranelift backends (feature flags).

**Core = syntax only, domains = semantics.** Each domain crate has its own:
- Value types and type system
- Function registry
- Self-contained backends (behind feature flags)

## Feature Flags

dew-core uses feature flags to manage complexity:

| Feature | Description |
|---------|-------------|
| (none)  | Basic expressions: numbers, variables, arithmetic |
| `cond`  | Conditionals: `if`/`then`/`else`, comparisons (`<`, `>=`, `==`), boolean logic (`and`, `or`, `not`) |
| `func`  | Function calls: `name(args...)` with extensible registry |

Domain crates automatically enable `func` (they rely on function calls).

## Quick Example

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{eval, scalar_registry};

// Parse an expression
let expr = Expr::parse("sin(x) + cos(y)").unwrap();

// Evaluate with variables
let mut vars = std::collections::HashMap::new();
vars.insert("x".to_string(), 0.5_f32);
vars.insert("y".to_string(), 1.0_f32);

let registry = scalar_registry();
let result = eval(expr.ast(), &vars, &registry).unwrap();
```

### With Conditionals

When dew-core is compiled with the `cond` feature:

```rust
use rhizome_dew_core::Expr;
use rhizome_dew_scalar::{eval, scalar_registry};

// Conditional expression
let expr = Expr::parse("if x > 0 then sqrt(x) else 0").unwrap();

let mut vars = std::collections::HashMap::new();
vars.insert("x".to_string(), 9.0_f32);

let result = eval(expr.ast(), &vars, &scalar_registry()).unwrap();
// result = 3.0
```

## Backends

Each domain crate includes three backends as optional features:

| Backend | Feature | Use case |
|---------|---------|----------|
| WGSL | `wgsl` | GPU shaders (WebGPU) |
| Lua | `lua` | Scripting, hot-reload |
| Cranelift | `cranelift` | Native JIT compilation |

Enable in `Cargo.toml`:

```toml
[dependencies]
rhizome-dew-scalar = { version = "0.1", features = ["wgsl", "lua", "cranelift"] }
```

## Crates

| Crate | Description |
|-------|-------------|
| `rhizome-dew-core` | Core AST and parsing (feature-gated conditionals and functions) |
| `rhizome-dew-cond` | Conditional backend helpers for domain crates |
| `rhizome-dew-scalar` | Scalar math: sin, cos, exp, lerp, etc. |
| `rhizome-dew-linalg` | Linear algebra: Vec2-4, Mat2-4, dot, cross, etc. |
| `rhizome-dew-complex` | Complex numbers: exp, log, polar, conjugate, etc. |
| `rhizome-dew-quaternion` | Quaternions: rotation, slerp, axis-angle, etc. |
