# WGSL Backend

Generate WebGPU Shading Language code from sap expressions.

## Enable

```toml
rhizome-sap-scalar = { version = "0.1", features = ["wgsl"] }
rhizome-sap-linalg = { version = "0.1", features = ["wgsl"] }
```

## sap-scalar

### Generate Expression

```rust
use rhizome_sap_core::Expr;
use rhizome_sap_scalar::wgsl::emit_wgsl;

let expr = Expr::parse("sin(x) + cos(y)").unwrap();
let wgsl = emit_wgsl(expr.ast()).unwrap();

println!("{}", wgsl.code);
// Output: sin(x) + cos(y)
```

### Generate Function

```rust
use rhizome_sap_scalar::wgsl::emit_wgsl_fn;

let expr = Expr::parse("x * x + y * y").unwrap();
let wgsl = emit_wgsl_fn("distance_squared", expr.ast(), &["x", "y"]).unwrap();

println!("{}", wgsl);
// Output:
// fn distance_squared(x: f32, y: f32) -> f32 {
//     return x * x + y * y;
// }
```

## sap-linalg

### Generate with Types

```rust
use rhizome_sap_core::Expr;
use rhizome_sap_linalg::wgsl::emit_wgsl;
use rhizome_sap_linalg::Type;
use std::collections::HashMap;

let expr = Expr::parse("normalize(v) * 2.0").unwrap();

// Declare variable types
let mut var_types: HashMap<String, Type> = HashMap::new();
var_types.insert("v".to_string(), Type::Vec3);

let result = emit_wgsl(expr.ast(), &var_types).unwrap();

println!("{}", result.code);
// Output: (normalize(v) * 2.0000000000)

println!("Result type: {:?}", result.typ);
// Output: Result type: Vec3
```

## Function Mapping

| sap | WGSL |
|-----|------|
| `lerp(a, b, t)` | `mix(a, b, t)` |
| `ln(x)` | `log(x)` |
| `log10(x)` | `(log(x) / log(10.0))` |
| `inversesqrt(x)` | `inverseSqrt(x)` |
| `x ^ y` | `pow(x, y)` |

Most functions map directly (sin, cos, exp, etc.).
