---
PLAN: "feat: QwenScheme — Qwen3.5 pre-tokenizer (NFC + regex split + byte level) and EncodeOrdinary, verified against the model's tokenizer.json"
TAG: v0.3.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 6089302139946885702
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> Independent. `webtyp/qwen` waits for this tag.

# Plan — `webtyp/tokenizer`: the Qwen3.5 scheme

## 0. Context

A tokenizer turns text into the token ids a model reads. This repository has one model-agnostic
BPE merge engine (`tokenizer.go`) and one **scheme** per model family (`scheme.go`): how text is
cut into *pre-tokens* before merging, and how tokens decode back to bytes. There are two today:
`ByteLevelScheme` (Granite) and `MetaspaceScheme` (bekko).

The first language model webtyp runs, **Qwen3.5**, needs a third one, `QwenScheme`. According to
its `tokenizer.json`, it works in three steps:

1. **Normalizer: NFC.** Characters are composed. For example `e` + U+0301 (combining acute)
   becomes `é` (U+00E9).
2. **Split** with this regular expression. The alternatives are tried **in order** at each
   position, and the first that matches wins:
   ```
   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   ```
3. **ByteLevel**: each piece's UTF-8 bytes are mapped with the GPT-2 byte-to-rune table, which
   `byte_level.go` already implements (`encodeBytes`, `runeToByte`).

Everything needed is in this repository:

| File | What it is |
|---|---|
| `testdata/qwen35_pretokenize.json` | 21 cases from the model's own tokenizer (Hugging Face `tokenizers`): `text`, `normalized` (after NFC), `pretokens` (after split + byte level), `ids` (full encoding, no special tokens) |
| `testdata/qwen35_tokenizer.json.gz` | the model's `tokenizer.json`: `model.vocab` (248 044 tokens), `model.merges` (strings `"a b"`), `added_tokens` |
| `nfc_latin.go` | **generated** table `nfcLatin` of 497 canonical compositions `{base, mark, precomposed}` for Latin letters (by `testdata/gen_nfc_latin.py`, do not edit) |
| `testdata/gen_*.py` | how the fixtures were produced (do not run them) |

## Development rules (inline)

- **Primary runtime: browser, TinyGo/WASM.** Every non-test file compiles under TinyGo.
- **Never import** in non-test files: `strings`, `fmt`, `errors`, `strconv` (use `webtyp.com/fmt`),
  `regexp` (it is large and slow under TinyGo; write the scanner by hand, as `pretokenize.go`
  does for Granite), `golang.org/x/text`, `sort`, `map[K]V` (a linear or binary search over a
  slice). `unicode` (stdlib) is allowed and already used, for `\p{L}`, `\p{M}`, `\p{N}` and
  `unicode.IsSpace` for `\s`.
- Do not change `tokenizer.go`, `ByteLevelScheme` or `MetaspaceScheme` behaviour. The existing
  Granite and bekko reference tests must keep passing unchanged.
- Tests may use `os`, `compress/gzip`, `encoding/json`, `reflect`. Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** Hugging Face `tokenizers` composes `normalizer → pre_tokenizer → model →
   decoder` per model. `tiktoken` ships one regex per encoding. llama.cpp has one
   `llm_tokenizer_*` pre-tokenizer per model family, selected by the GGUF `tokenizer.ggml.pre`
   key (here: `qwen35`). All three make the pre-tokenizer a per-family choice over one engine.
   Our `Scheme` is that choice.
2. **Novice-name test.** `tokenizer.QwenScheme{}` next to `ByteLevelScheme{}` and
   `MetaspaceScheme{}`, used as `tokenizer.New(tokenizer.Config{…, Scheme: tokenizer.QwenScheme{}})`.
3. **Complexity ledger.** Concepts +2 (`QwenScheme`, `EncodeOrdinary`) / −0. Ways to do the same thing +0 (`Encode` is built on `EncodeOrdinary`).
4. **Where it belongs.** Pre-tokenization is the tokenizer's job. Special tokens, the chat
   template and ids like `<|im_start|>` belong to `webtyp/qwen`.
5. **What it deletes.** Nothing. This is new capability.

## Stage 1 — NFC for Latin text (`nfc.go`)

`func composeLatin(text string) string`. Walk the runes. When a rune in U+0300–U+036F follows a
rune `b`, and `(b, mark)` is in `nfcLatin`, replace both with the precomposed rune, and keep
composing the result with the next mark (this handles stacked marks such as `A` + U+0302 + U+0301
→ `Ấ`). Otherwise keep the runes as they are. Look up `nfcLatin` with a binary search on
`(base, mark)` (the table is sorted). Return `text` unchanged, without allocating, when it has no
rune in U+0300–U+036F.

This is NFC restricted to Latin letters, which covers Spanish, Portuguese, French and the rest of
the Latin script. Document that limitation in the doc comment of `QwenScheme`.

## Stage 2 — the split (`qwen_pretokenize.go`)

`func qwenSplit(runes []rune) [][]rune`, a hand-written scanner equivalent to the regex above.
At position `i`, try in order:

1. **Contraction**: `'` followed by (case-insensitive) `s`, `t`, `re`, `ve`, `m`, `ll`, `d`.
2. **Word**: optionally one rune that is not `\r`, `\n`, a letter (`unicode.IsLetter`) or a number
   (`unicode.IsNumber`), then one or more letters or marks (`unicode.IsMark`). It must contain
   at least one letter or mark.
3. **Number**: exactly one rune that `unicode.IsNumber`. Digits are split one by one.
4. **Punctuation**: an optional single space `' '`, then one or more runes that are not
   whitespace, letter, mark or number, then any `\r` / `\n` runes.
5. **Newlines**: optional whitespace, then one or more `\r` / `\n`. Match the longest run of
   whitespace that ends with a newline rune.
6. **Trailing whitespace**: `\s+(?!\S)`. Take the maximal whitespace run. If it is followed by a
   non-space rune and is longer than 1, give back its last rune (the run ends one before). If it
   has length 1 and is followed by a non-space rune, this alternative fails.
7. **Whitespace**: `\s+`, the maximal run.

If nothing matches (impossible for valid input), consume one rune.

`QwenScheme.Pretokenize(text)` = `composeLatin` → `qwenSplit` → `encodeBytes` of each piece's UTF-8.
`ByteFallbackSymbol` and `DecodeToken` are the same as `ByteLevelScheme`'s. Call the existing
functions and do not copy them.

## Stage 2b — encoding without special tokens (`tokenizer.go`)

`Encode` always wraps the text in `BosTokenID … EosTokenID`. That is right for the embedding
models, but a language-model prompt must not: Qwen3.5 has no BOS, and its prompt must not end in
EOS. The adapter (`webtyp/qwen`) inserts special tokens such as `<|im_start|>` itself, by id.
Add:

```go
// EncodeOrdinary converts text into token ids with no special tokens around it.
func (t *BPE) EncodeOrdinary(dst []int32, text string) []int32
```

and make `Encode` call it: `append(bos)`, then `EncodeOrdinary`, then `append(eos)`. The pretoken
loop exists once. `EncodeOrdinary` is tiktoken's name for exactly this (`encode_ordinary`).

## Stage 3 — tests (`qwen_test.go`)

| Test | Asserts |
|---|---|
| `TestQwen_Normalize` | for every case, `composeLatin(text) == normalized` |
| `TestQwen_Pretokenize` | for every case, `QwenScheme{}.Pretokenize(text)` equals `pretokens` exactly (use `reflect.DeepEqual`; for an empty text both are empty) |
| `TestQwen_EncodeMatchesReference` | load `testdata/qwen35_tokenizer.json.gz`; build the vocab slice from `model.vocab` plus `added_tokens` at their ids (length = max id + 1); merges from `model.merges`; `tokenizer.New(Config{Vocab, Merges, Scheme: QwenScheme{}})` for every case `EncodeOrdinary(nil, text)` equals `ids` |
| `TestQwen_DecodeRoundTrip` | for every case, `Decode(EncodeOrdinary(nil, text))` equals `normalized` |
| `TestEncode_IsBosOrdinaryEos` | with the Granite fixture, `Encode(nil, s)` equals `[bos] + EncodeOrdinary(nil, s) + [eos]` |
| `TestComposeLatin_NoMarksNoAlloc` | `testing.AllocsPerRun` of `composeLatin("hola mundo")` is 0 |

If `TestQwen_EncodeMatchesReference` fails while `TestQwen_Pretokenize` passes, the difference is
in how the test builds the vocabulary (added tokens), not in the scheme. Read `Config` in
`tokenizer.go` before changing anything else.

## Stage 4 — docs

`README.md`: add `QwenScheme` to the list of schemes (model, normalizer, split, byte level),
and state the NFC limitation. `AGENTS.md`: add `regexp` and `golang.org/x/text` to the list of
imports never to use, if they are not there yet.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `nfc.go` | `TestQwen_Normalize`, `TestComposeLatin_NoMarksNoAlloc` |
| 2 | `qwen_pretokenize.go`, `scheme` doc | `TestQwen_Pretokenize` |
| 2b | `tokenizer.go` | `TestEncode_IsBosOrdinaryEos`; existing reference tests unchanged |
| 3 | `qwen_test.go` | all tests pass under `gotest` and `gotest -tinygo`; the Granite and bekko reference tests unchanged |
| 4 | `README.md`, `AGENTS.md` | `grep -n QwenScheme README.md` |
