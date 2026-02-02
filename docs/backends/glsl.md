# GLSL Backend

Generate OpenGL Shading Language code from wick expressions.

## Enable

```toml
wick-scalar = { version = "0.1", features = ["glsl"] }
wick-linalg = { version = "0.1", features = ["glsl"] }
```

## wick-scalar

### Generate Expression

```rust
use wick_core::Expr;
use wick_scalar::glsl::emit_glsl;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let glsl = emit_glsl(expr.ast()).unwrap();

println!("{}", glsl.code);
// Output: sin(x) + cos(y)
```

### Generate Function

```rust
use wick_scalar::glsl::emit_glsl_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let glsl = emit_glsl_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", glsl);
// Output:
// float distance_squared(float x, float y) {
//     return x * x + y * y;
// }
```

## wick-linalg

### Generate with Types

```rust
use wick_core::Expr;
use wick_linalg::glsl::emit_glsl;
use wick_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("normalize(v) * 2.0").unwrap();

// Declare variable types
let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("v".to_string(), Type::Vec3);

let result = emit_glsl(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: (normalize(v) * 2.0000000000)

println!("Result type: {:?}", result.typ);
// Output: Result type: Vec3
```

## Function Mapping

| wick | GLSL |
|-----|------|
| `lerp(a, b, t)` | `mix(a, b, t)` |
| `ln(x)` | `log(x)` |
| `log10(x)` | `(log(x) / log(10.0))` |
| `inversesqrt(x)` | `inversesqrt(x)` |
| `x ^ y` | `pow(x, y)` |
| `fract(x)` | `fract(x)` |

Most functions map directly (sin, cos, exp, etc.).

## GLSL Versions

The generated code is compatible with:
- GLSL 4.50+ (OpenGL 4.5+)
- GLSL ES 3.00+ (OpenGL ES 3.0+, WebGL 2.0)

## Comparison with WGSL

| Feature | GLSL | WGSL |
|---------|------|------|
| Target | OpenGL, Vulkan, WebGL 2 | WebGPU |
| Type syntax | `vec3`, `mat4` | `vec3<f32>`, `mat4x4<f32>` |
| Function prefix | None | None |
| Naming | `inversesqrt` | `inverseSqrt` |

Both backends generate functionally equivalent code with minor syntax differences.
