# Introduction

Sap is a minimal expression language designed for code generation across multiple backends.

## Goals

- **Minimal**: Just functions and numeric values
- **Multi-target**: WGSL, Cranelift, Lua backends
- **Composable**: Build complex expressions from simple primitives

## Architecture

```
sap-core         # Core expression AST and types
├── sap-wgsl     # WGSL code generation
├── sap-cranelift # Cranelift JIT compilation
└── sap-lua      # Lua code generation
```
