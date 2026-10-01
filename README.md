# tokenizer
<img src="docs/img/badges.svg">

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

## Schemes

- **`ByteLevelScheme`**: Granite models (pretokenizer regex + GPT-2 byte mapping).
- **`MetaspaceScheme`**: Bekko models (SentencePiece metaspace `▁` + byte fallback).
- **`QwenScheme`**: Qwen3.5 models (NFC restricted to Latin script + Qwen3.5 regex split + GPT-2 byte mapping).
- **`Lfm2Scheme`**: LFM2 models (LiquidAI LFM2.5 pretokenizer regex + GPT-2 byte mapping).

A scheme says whether it may skip merges (`IgnoreMerges`), as the model's tokenizer.json does.

## Streaming generated text

A byte-level token can end in the middle of a UTF-8 character. `Stream` holds that tail back:

```go
var s tokenizer.Stream
chunk := s.Write(scheme.DecodeToken(nil, vocab[id])) // "" while a character is incomplete
// … at the end: s.Flush(); s.Text() is the whole answer
```

## Loading a model's tokenizer

`webtyp/weightsc` writes the vocabulary into the `.wtypw` artifact and the merges into a companion
`.merges` file. `ParseMerges` turns that file into `Config.Merges`:

```go
bpe, err := tokenizer.New(tokenizer.Config{
	Scheme: tokenizer.Lfm2Scheme{},
	Vocab:  artifact.Tokenizer.Vocab,
	Merges: tokenizer.ParseMerges(mergesFile),
})
```
