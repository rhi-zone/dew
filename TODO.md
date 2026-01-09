# TODO

## Backlog

### Port from resin-expr

- [x] Port `resin-expr` core expression types and AST
- [x] Port `resin-expr-std` standard library functions (excluding noise)

### Core

- [x] Define expression AST (functions, numeric values)
- [ ] Implement type system for expressions
- [ ] Add expression validation/normalization

### Backends

- [ ] WGSL code generation
- [ ] Cranelift JIT compilation
- [ ] Lua code generation (maybe)

### Infrastructure

- [x] Set up CI tests for all backends
- [ ] Add integration tests
- [ ] Documentation examples
