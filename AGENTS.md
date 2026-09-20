# Agent Guide — `webtyp/tokenizer`

Constraints for agents working on this library. **Read this before any change.**
The current work order is [docs/PLAN.md](docs/PLAN.md); the master index is
[`agent/docs/MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md).

---

## What this library is

Text in, token ids out. Nothing else — no embeddings, no model weights, no storage. It owns
byte-level BPE tokenization exactly matching `granite-embedding-97m-multilingual-r2`'s real
`tokenizer.json`. Its **primary runtime is a browser tab compiled with TinyGo**. The host
(`go test`) is a development convenience, not the target.

## Dependencies: `webtyp.com/fmt`. Nothing else.

No `weights`, no `transformer`, no `syscall/js`, no `regexp` in the shipped code (see
`docs/PLAN.md` for why the reference regex can't run under Go's RE2 engine anyway). This
package receives a `Config{Vocab, Merges}` built by the caller — it never fetches or decodes
an artifact itself.

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
```

`gotest` alone passing means nothing here. TinyGo's stdlib is a **subset** of what
`GOOS=js GOARCH=wasm go build` accepts — a package can compile for one and fail the other.

## Never import these

| Never | Use instead |
|---|---|
| `fmt`, `errors`, `strconv`, `strings` | `webtyp.com/fmt` |
| `encoding/json`, `net/http`, `context` (stdlib), `os`, `log` | nothing here needs them |
| `regexp` | a hand-written scanner (see `docs/PLAN.md` — the reference pretokenizer regex uses a negative lookahead RE2 can't run anyway) |
| `syscall/js` | this package never touches the DOM |
| `map[K]V` | a sorted slice + `sort.Search` (binary search) — see `docs/PLAN.md` for why a linear scan over 180k vocab entries is the wrong call here |

`unicode` (stdlib) is fine and expected — its category tables (`unicode.L`, `unicode.Lu`,
`unicode.M`, ...) are what replaces the `\p{...}` classes in the reference regex, and they do
compile under TinyGo.

## Numeric/correctness rules that cost hours when learned late

- **The tokenizer is a port, not a design.** The reference is the real `tokenizer.json` of
  `ibm-granite/granite-embedding-97m-multilingual-r2`, not a general BPE paper. When in doubt,
  the fixture in `testdata/reference_cases.json` (generated from 🤗 `tokenizers` itself) is the
  ground truth — never hand-guessed expected ids.
- **`normalizer` is `null`.** No lowercasing, no accent stripping, no NFKC. Don't add any of
  them "for robustness" — it changes every token id downstream.
- **`ignore_merges: true` is not optional.** Skipping it doesn't break correctness by much but
  makes every common word run the full merge loop — implement the whole-word vocab shortcut.
- **`add_prefix_space` matters for the first pretoken of a string.** Verify it against
  `tokenizer.json` rather than assuming true or false.

## Layout & tests

- Flat hierarchy: Go files in the repo root. No subdirectories for library code except
  `testdata/`.
- Max 500 lines per file; split by domain (pretokenizer, byte-level codec, BPE merge, vocab
  lookup) and rename when exceeded.
- Publish with `gopush 'message'` — never `git commit`/`git push` directly.

## Common mistakes to avoid

- Assuming this is WordPiece (a previous, now-deleted version of this plan did — the real
  model uses byte-level BPE; don't resurrect that assumption from an old branch or memory).
  - Assuming `\s+(?!\S)` needs a regex engine with lookahead support. It doesn't — see
  `docs/PLAN.md` for the equivalent scan.
  - Adding a `Merges` field to `weights.TokenizerConfig` in the `webtyp/weights` repo to carry
  merge rules through the artifact. That's published API in a different repo; this plan
  deliberately keeps merges in a separate `.merges` file instead.
