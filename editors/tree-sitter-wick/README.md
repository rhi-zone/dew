# tree-sitter-wick

Tree-sitter grammar for the Wick expression language.

## Usage

### Generate Parser

```bash
npm install
npm run generate
```

### Test

```bash
npm test
```

### Editor Integration

#### Neovim

Add to your nvim-treesitter config:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.wick = {
  install_info = {
    url = "https://github.com/rhi-zone/wick",
    files = { "editors/tree-sitter-wick/src/parser.c" },
    location = "editors/tree-sitter-wick",
  },
  filetype = "wick",
}
```

Then copy `queries/highlights.scm` to `~/.config/nvim/queries/wick/highlights.scm`.

#### Helix

Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "wick"
scope = "source.wick"
file-types = ["wick"]
roots = []
comment-token = "//"

[[grammar]]
name = "wick"
source = { git = "https://github.com/rhi-zone/wick", subpath = "editors/tree-sitter-wick" }
```

## Syntax

```wick
// Basic arithmetic
x * 2 + y

// Function calls
sin(x) + cos(y)

// Conditionals
if x > 0 then sqrt(x) else 0

// Boolean logic
x > 0 and y < 10 or not z
```
