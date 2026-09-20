# tokenizer

byte-level BPE tokenizer for `granite-embedding-97m-multilingual-r2`: text in, token ids out.

## Targets & Build Verification

| Target / Tool | Command | Result |
|---|---|---|
| Native Go test | `go test .` | **PASS** (100% reference match) |
| Go WASM build | `GOOS=js GOARCH=wasm go build ./...` | **PASS** |
| TinyGo build / gotest | `gotest -tinygo` / `tinygo build` | *N/A (tinygo not installed in sandbox environment)* |

## Strict Constraints Verified

- `map[K]V` in shipped code: **None** (uses sorted slices + binary search lookup).
- Prohibited stdlib imports (`fmt`, `errors`, `regexp`, `strings`): **None** (zero dependencies outside `unicode` stdlib).
