---
PLAN: "feat: webtyp/tokenizer — pluggable Scheme (ByteLevel/Granite + Metaspace/Bekko), fix the Alt1/Alt2 backtracking bug"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 11874997801151083228
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus comentarios
> en inglés.

# Plan — `webtyp/tokenizer`, `Scheme` plegable + corrección de Alt1/Alt2

## Por qué existe este plan

El repo ya tiene una implementación completa y en su mayoría correcta del BPE byte-level de
`granite-embedding-97m-multilingual-r2` (PR #1, mergeado). Dos problemas la obligan a
cambiar, no a parchearse:

1. **El proyecto acaba de cambiar de modelo, a `bekko-embedding-v1-a8m`**
   (`agent/docs/SMALL_MODEL_FOR_EMBEDING.md` §6, ya lo nombraba como plan B desde el
   17/09). Bekko usa un tokenizer **completamente distinto** — Metaspace + byte-fallback en
   vez de la regex byte-level de Granite — verificado directo contra su `tokenizer.json` real
   (no de memoria, ver abajo).
2. **Una revisión encontró un bug real en `matchAlt12`** (`pretokenize.go`): no hace
   backtracking, así que un carácter que pertenece a las dos clases de la regex a la vez (Lo,
   Lm, M — letras "otras" como CJK, árabe, devanagari) se consume mal cuando está pegado a una
   letra mayúscula latina. Verificado contra el módulo `regex` de Python (que sí soporta
   lookahead, a diferencia de RE2/Go `regexp`), con el patrón real del `tokenizer.json`.

**La corrección de fondo, para las dos cosas a la vez:** el pretokenizador y el decoder no
son un solo algoritmo fijo — son un **esquema** que varía por modelo, exactamente como
`transformer.Config` ya varía por modelo (capas, cabezales, pooling). Esta librería nunca
tuvo esa abstracción; se escribió hardcodeada a un solo modelo. Este plan la introduce, y dos
implementaciones concretas la usan: la de Granite (existente, corregida) y la de Bekko
(nueva). El motor de merges BPE (`encodePretoken`'s loop principal) ya era genérico — **no se
toca**, más allá de la rama de byte-fallback nueva que necesita (ver Cambio 4).

## Design gate

**1. Prior art.** La librería de referencia, 🤗 `tokenizers` (Rust/Python), tiene exactamente
esta forma: `PreTokenizer` y `Decoder` son componentes plegables, seleccionados por modelo
(`ByteLevel`, `Metaspace`, `WordPiece`, ...) — este plan port esa arquitectura, no inventa una
nueva. Los nombres de los dos tipos concretos de abajo (`ByteLevelScheme`,
`MetaspaceScheme`) son literalmente los nombres que usa `tokenizer.json` bajo `pre_tokenizer.type`
y `decoder.type`. Segundo precedente, interno: `webtyp/storage` (puerto + backends
intercambiables) ya es el patrón que usa este ecosistema para "una operación, varias
implementaciones concretas".

**2. Novice-name test.** Un desarrollador que abrió un `tokenizer.json` alguna vez reconoce
`ByteLevelScheme`/`MetaspaceScheme` sin documentación — son los nombres reales de
HuggingFace. `Config.Scheme` no necesita explicación: es obvio que hay que elegir uno.

**3. Complexity ledger.**
```
Conceptos nuevos para el desarrollador      +1 (interfaz Scheme; las dos implementaciones ya
                                                se conocen por nombre si se leyó un tokenizer.json)
Formas de tokenizar soportadas               2 (antes: 1, hardcodeada a Granite) — exactamente
                                                las 2 que usan los candidatos vivos de D5
Repos a tocar                                0 (todo interno a este repo)
```

**4. Dónde vive.** Acá, no en `embed`: pretokenización y decode son el trabajo central de
`tokenizer`, no una responsabilidad que `embed` deba absorber.

**5. Qué borra.** Las constantes de paquete `BosTokenID`, `PadTokenID`, `EosTokenID`
(hardcodeadas a los ids de Granite: 179934/179935/179938) — pasan a ser campos de `Config`,
porque Bekko usa ids completamente distintos (`bos_token_id: 2`, `pad_token_id: 0`,
`eos_token_id: 1` — verificado en su `config.json` real). Mantener las constantes globales
mientras se agrega un segundo modelo sería el mismo error que ya se corrigió en `transformer`
(D8 nota (i) del índice maestro): un valor de un modelo, aplicado en silencio a otro.

## Verificado contra el modelo real, no de memoria

### Granite — la corrección de `matchAlt12`

El bug: `isUpperLike` (clase X: `Lu,Lt,Lm,Lo,M`) e `isLowerLike` (clase Y: `Ll,Lm,Lo,M`) **se
superponen** en `Lm,Lo,M`. La implementación actual consume todo lo que matchea X primero, sin
nunca reconsiderar — así que un carácter Lo pegado a una mayúscula Latina queda mal
clasificado. La regex real (Alt1 `X*Y+`, Alt2 `X+Y*`, backtracking, probado con el módulo
`regex` de Python contra el patrón real citado en la sección "Fuera de alcance" de abajo):

```
'你A'   -> ['你', 'A']      (no ['你A'])
'A你B'  -> ['A你', 'B']     (no ['A你B'] ni ['A', '你B'])
'你好世界' -> ['你好世界']    (uno solo — ningún carácter es Lu/Lt, todo Lo)
'Hello World' -> ['Hello', ' World']   (sin cambios, ya funcionaba)
```

**Reemplazá el cuerpo de `matchAlt12`** (el bloque entre el cálculo de `hasPrefix`/`curr` — que
NO cambia — y el `return 0` final) por esto, que sí hace el backtracking que la regex hace:

```go
func matchAlt12(runes []rune, i, n int) int {
	curr := i
	if curr < n {
		r := runes[curr]
		if !isCRLF(r) && !isLetter(r) && !isNumber(r) {
			curr++
		}
	}
	j := curr

	// maxX = longest run of X-class runes (Lu, Lt, Lm, Lo, M) starting at j.
	maxX := 0
	for j+maxX < n && isUpperLike(runes[j+maxX]) {
		maxX++
	}

	// Alt1: X*Y+, tried first, greedy-then-backtrack. Find the LARGEST prefixLen in
	// [0, maxX] such that the rune right after it is Y-class (Ll, Lm, Lo, M) — that is
	// exactly what a backtracking regex engine converges on for X*Y+.
	for prefixLen := maxX; prefixLen >= 0; prefixLen-- {
		pos := j + prefixLen
		if pos < n && isLowerLike(runes[pos]) {
			yCount := 0
			for pos+yCount < n && isLowerLike(runes[pos+yCount]) {
				yCount++
			}
			matchedLen := (j - i) + prefixLen + yCount
			matchedLen += matchContraction(runes, i+matchedLen, n)
			return matchedLen
		}
	}

	// Alt2: X+Y* — only reached when Alt1 found no Y-class rune anywhere in [j, j+maxX].
	// That means the Y* trailing part is necessarily empty too (same position was checked
	// and failed), so the match is exactly the X run.
	if maxX >= 1 {
		matchedLen := (j - i) + maxX
		matchedLen += matchContraction(runes, i+matchedLen, n)
		return matchedLen
	}

	return 0
}
```

Borrá la variable `hasPrefix` (ya no se usa — la versión vieja la dejaba sin usar, `_ =
hasPrefix`; la versión nueva ni la declara).

**No re-derives esto de memoria ni confíes en la prosa de arriba como especificación
completa**: el test `TestPretokenizeAgainstReference` (existente, sin tocar) ya corre contra
`testdata/pretokenize_reference.json`, que es la referencia real. Agregá estos 3 objetos al
final del array de ese archivo — son las 3 entradas exactas, ya codificadas byte-level
(`encodeBytes`), generadas corriendo la tabla real de `byte_level.go` — no las re-derives a
mano, copialas tal cual:

```json
  {
    "input": "你A",
    "want": [
      "ä½ł",
      "A"
    ]
  },
  {
    "input": "A你B",
    "want": [
      "Aä½ł",
      "B"
    ]
  },
  {
    "input": "你好世界",
    "want": [
      "ä½łå¥½ä¸ĸçķĮ"
    ]
  }
```

**Estos 3 casos fallan contra la implementación actual (sin el fix de arriba) — es
intencional, es la prueba de que el fixture ejercita el bug real.** Con el fix aplicado, los
tres pasan.

### Bekko — el esquema `Metaspace`, verificado contra su `tokenizer.json` real

`hotchpotch/bekko-embedding-v1-a8m` (HuggingFace, MIT). Su `tokenizer.json` real (no
inferido):

```
"normalizer":     {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
"pre_tokenizer":  {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}
"model":          {"type": "BPE", "byte_fallback": true, "ignore_merges": true, "vocab": {...256000 entradas...}, "merges": [...]}
"decoder":        {"type": "Sequence", "decoders": [
                     {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                     {"type": "ByteFallback"},
                     {"type": "Fuse"}
                   ]}
```

`▁` es U+2581 (LOWER ONE EIGHTH BLOCK). El vocab tiene 255 entradas de byte-fallback,
formato exacto `<0x00>` .. `<0xFF>` (mayúsculas, dos dígitos hex) — confirmado listando el
vocab real, no asumido.

**Casos reales**, generados corriendo el `tokenizer.json` real con 🤗 `tokenizers` (instalalo
en un venv para reproducir: `pip install tokenizers`, `Tokenizer.from_file(...)`) — estos son
el fixture nuevo, no los inventes a mano:

```
'hola mundo'                -> pretokens: ['▁hola', '▁mundo']
'Hola Mundo'                -> pretokens: ['▁Hola', '▁Mundo']
'  espacios   múltiples  '  -> pretokens: ['▁', '▁espacios', '▁', '▁', '▁múltiples', '▁', '▁']
'café niño mañana'          -> pretokens: ['▁café', '▁niño', '▁mañana']
'你好世界'                    -> pretokens: ['▁你好世界']   (una corrida sin espacios: UN pretoken;
                                                        el desglose a ['▁你','好','世界'] ocurre
                                                        en el merge loop existente, no acá)
'你A'                        -> pretokens: ['▁你A']
'a'                          -> pretokens: ['▁a']
' a'                         -> pretokens: ['▁a']          (el prepend NO duplica el ▁)
'a '                         -> pretokens: ['▁a', '▁']
'😀 emoji test'               -> pretokens: ['▁😀', '▁emoji', '▁test']
''                           -> pretokens: nil
```

**Algoritmo — 3 pasos, ningún regex, ninguna clase Unicode:**

1. Normalizar: cada `' '` (space, U+0020) del texto de entrada se reemplaza por `'▁'`.
2. Prepend: si el texto ya empieza con `' '` (que el paso 1 ya convirtió a `'▁'`), no hagas
   nada más — ya hay un `▁` al principio. Si NO, insertá un `'▁'` al principio (esto es
   `prepend_scheme: "always"`, que en la práctica es "asegurate de que haya exactamente un
   `▁` al inicio, sin duplicarlo si ya está").
3. Split: partí el resultado en pretokens, uno por cada `'▁'` — cada `▁` arranca un pretoken
   nuevo que sigue hasta el próximo `▁` (exclusive) o el final de la cadena.

```go
const metaspaceChar = '▁' // U+2581

// MetaspaceScheme is bekko-embedding-v1-a8m/a25m's pretokenizer + decoder: SentencePiece-style
// Metaspace (prepend_scheme=always, split=true) plus byte_fallback for out-of-vocab runes.
// Verified against the real tokenizer.json — see docs/LAST_PLAN_EXECUTED.md.
type MetaspaceScheme struct{}

func (MetaspaceScheme) Pretokenize(text string) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}

	normalized := make([]rune, 0, len(runes)+1)
	if runes[0] != ' ' {
		normalized = append(normalized, metaspaceChar)
	}
	for _, r := range runes {
		if r == ' ' {
			normalized = append(normalized, metaspaceChar)
		} else {
			normalized = append(normalized, r)
		}
	}

	var tokens []string
	start := -1
	for idx, r := range normalized {
		if r == metaspaceChar {
			if start != -1 {
				tokens = append(tokens, string(normalized[start:idx]))
			}
			start = idx
		}
	}
	if start != -1 {
		tokens = append(tokens, string(normalized[start:]))
	}
	return tokens
}
```

**Byte fallback, verificado con un carácter genuinamente fuera de vocabulario** (U+0870,
árabe extendido — no está en el vocab de 256k):

```
'ࡰ' (U+0870, 3 bytes UTF-8: 0xE0 0xA1 0xB0)
  -> pretoken: ['▁ࡰ']
  -> símbolos iniciales del merge loop (Cambio 4): ['▁', '<0xE0>', '<0xA1>', '<0xB0>']
  -> ids finales, texto decodificado: ' ࡰ' (el espacio viene de decodificar '▁')
```

```go
func (MetaspaceScheme) ByteFallbackSymbol(b byte) (string, bool) {
	const hexDigits = "0123456789ABCDEF"
	return string([]byte{'<', '0', 'x', hexDigits[b>>4], hexDigits[b&0x0F], '>'}), true
}

func (MetaspaceScheme) DecodeToken(dst []byte, tok string) []byte {
	if b, ok := parseByteFallbackToken(tok); ok {
		return append(dst, b)
	}
	for _, r := range tok {
		if r == metaspaceChar {
			dst = append(dst, ' ')
		} else {
			dst = append(dst, []byte(string(r))...)
		}
	}
	return dst
}

func parseByteFallbackToken(tok string) (byte, bool) {
	if len(tok) != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>' {
		return 0, false
	}
	hi, ok1 := hexDigitVal(tok[3])
	lo, ok2 := hexDigitVal(tok[4])
	if !ok1 || !ok2 {
		return 0, false
	}
	return hi<<4 | lo, true
}

func hexDigitVal(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}
```

Poné todo lo de arriba en un archivo nuevo, `metaspace.go`.

## Cambio 1 — la interfaz `Scheme`, en un archivo nuevo `scheme.go`

```go
// Scheme is the model-specific pretokenization/decode behavior a Config selects. The BPE
// merge engine in tokenizer.go is model-agnostic and never changes; only the scheme does.
type Scheme interface {
	// Pretokenize splits text into pretoken chunks in this model's own alphabet — already
	// byte-remapped (ByteLevelScheme) or literal UTF-8 (MetaspaceScheme).
	Pretokenize(text string) []string
	// ByteFallbackSymbol returns the vocab-lookup string this scheme uses to represent one
	// raw byte, for a rune that has no vocab entry of its own. ok is false for a scheme
	// that has no byte-fallback mechanism (ByteLevelScheme: every byte is already a vocab
	// entry by construction, so this is never called in practice).
	ByteFallbackSymbol(b byte) (symbol string, ok bool)
	// DecodeToken appends tok's raw bytes to dst.
	DecodeToken(dst []byte, tok string) []byte
}
```

## Cambio 2 — `ByteLevelScheme`, envolviendo el código de Granite ya existente

En `byte_level.go`, agregá (no borres nada de lo que ya hay — `byteToRune`, `runeToByte`,
`encodeBytes` se quedan igual, `ByteLevelScheme` los llama):

```go
// ByteLevelScheme is granite-embedding-97m-multilingual-r2's pretokenizer: the reference
// regex (pretokenize.go) plus GPT-2 byte-level encode/decode. See docs/LAST_PLAN_EXECUTED.md.
type ByteLevelScheme struct{}

func (ByteLevelScheme) Pretokenize(text string) []string {
	return pretokenize(text)
}

func (ByteLevelScheme) ByteFallbackSymbol(b byte) (string, bool) {
	return "", false
}

func (ByteLevelScheme) DecodeToken(dst []byte, tok string) []byte {
	for _, r := range tok {
		if b, ok := runeToByte(r); ok {
			dst = append(dst, b)
		} else {
			dst = append(dst, []byte(string(r))...)
		}
	}
	return dst
}
```

`pretokenize()` en `pretokenize.go` sigue existiendo tal cual (con el fix de Alt1/Alt2 de
arriba) — `ByteLevelScheme.Pretokenize` es un envoltorio de una línea, no una reescritura.

## Cambio 3 — `Config` gana `Scheme` y los tres ids especiales

En `tokenizer.go`, **borrá** las tres constantes de paquete:

```go
// BORRAR estas tres líneas:
const (
	BosTokenID int32 = 179934
	PadTokenID int32 = 179935
	EosTokenID int32 = 179938
)
```

Reemplazá `Config` por:

```go
// Config is the data a caller loads once per model.
type Config struct {
	// Vocab[id] is this scheme's token string for that id. Index IS the id.
	Vocab []string
	// Merges, in rank order (rank = index). Each entry is "left right".
	Merges []string
	// Scheme selects the pretokenizer/decoder for this model. Required — there is no
	// default, because a silent default is exactly how a Granite id ends up applied to a
	// Bekko vocab (see Design gate §5).
	Scheme Scheme
	// BosTokenID, EosTokenID, PadTokenID are this model's special token ids (verify in its
	// real config.json — do NOT reuse granite's 179934/179938 for another model).
	BosTokenID int32
	EosTokenID int32
	PadTokenID int32
}
```

`BPE` gana un campo `scheme Scheme` y los tres ids, poblados en `New`. `New` valida
`cfg.Scheme != nil` (importá `"webtyp.com/fmt"` para el error — el único import nuevo del
repo; el resto sigue prohibido, ver `AGENTS.md`):

```go
func New(cfg Config) (*BPE, error) {
	if cfg.Scheme == nil {
		return nil, fmt.Err("tokenizer: Config.Scheme is required")
	}
	// ... el resto de New no cambia — sigue armando sortedVocab/sortedMerges igual ...
	return &BPE{
		vocab:        cfg.Vocab,
		sortedVocab:  sortedVocab,
		sortedMerges: sortedMerges,
		scheme:       cfg.Scheme,
		bosTokenID:   cfg.BosTokenID,
		eosTokenID:   cfg.EosTokenID,
		padTokenID:   cfg.PadTokenID,
	}, nil
}
```

`Encode` usa `t.bosTokenID`/`t.eosTokenID` en vez de las constantes borradas, y
`t.scheme.Pretokenize(text)` en vez de llamar a `pretokenize(text)` directo. `Decode` recorre
`t.vocab[id]` y llama a `t.scheme.DecodeToken(bytes, tokenStr)` en vez del loop de
`runeToByte` que tenía adentro; sigue saltando `id == t.bosTokenID || id == t.eosTokenID ||
id == t.padTokenID`.

## Cambio 4 — `encodePretoken` gana la rama de byte-fallback

Es el único cambio al motor de merges, y solo dispara cuando un rune no tiene entrada de
vocab propia (con Granite esto no pasa nunca en la práctica — su alfabeto byte-remapeado
garantiza que los 256 valores base están en vocab; con Bekko sí pasa, para los caracteres
genuinamente fuera de las 256k entradas).

Reemplazá el bloque que arma `symbols` a partir de `runes`:

```go
// ANTES:
symbols := make([]string, len(runes))
for i, r := range runes {
	symbols[i] = string(r)
}
```

```go
// DESPUÉS:
symbols := make([]string, 0, len(runes))
for _, r := range runes {
	s := string(r)
	if _, ok := t.lookupVocab(s); ok {
		symbols = append(symbols, s)
		continue
	}
	fellBack := false
	for _, b := range []byte(s) {
		if sym, ok := t.scheme.ByteFallbackSymbol(b); ok {
			symbols = append(symbols, sym)
			fellBack = true
		}
	}
	if !fellBack {
		symbols = append(symbols, s)
	}
}
```

El resto de `encodePretoken` (el loop de merges, `lookupMergeRank`, el mapeo final a ids) **no
cambia**.

## Tests

**Los fixtures de Bekko ya están generados y commiteados — no los regeneres.** Este mismo plan
los trae:

- `testdata/pretokenize_reference.json`: **todavía no tiene** los 3 casos nuevos de Granite —
  quedaron afuera del commit a propósito, porque hoy fallan (ver la sección de arriba) y este
  repo no publica con tests rojos. Agregalos vos, con el bloque JSON exacto de la sección
  "Granite — la corrección de `matchAlt12`" de arriba, como parte de este mismo cambio.
- `testdata/bekko_reference_cases.json`: 10 casos reales, formato `{"text":..., "ids":[...]}`
  igual que `reference_cases.json`, generados corriendo el `tokenizer.json` real de
  `hotchpotch/bekko-embedding-v1-a8m` con 🤗 `tokenizers` — `add_special_tokens=True`, así que
  `ids` ya incluye `2` (BOS) al principio y `1` (EOS) al final, tal cual `Encode` los va a
  producir.
- `testdata/bekko_tokenizer_config.json.gz`: el vocab completo (256 000 entradas, denso, sin
  huecos — confirmado) y las 580 604 merges de Bekko, mismo formato `{"vocab":[...],
  "merges":[...]}` que `tokenizer_config.json.gz` usa para Granite. Cargalo con el mismo
  patrón que `loadTestBPE` en `tokenizer_test.go`, pasándole `Scheme: MetaspaceScheme{}` y
  `BosTokenID: 2, EosTokenID: 1` (Bekko real, de su `config.json`).

| Test | Verifica |
|---|---|
| `TestPretokenizeAgainstReference` (existente) | los 3 casos nuevos de Granite ya están en el fixture — el test no cambia, solo corre sobre más datos |
| `TestMetaspacePretokenize` (nuevo) | los 11 casos de la sección Bekko de arriba, contra `MetaspaceScheme{}.Pretokenize` directo (no hace falta el vocab de 256k para este test — son solo el algoritmo de split) |
| `TestEncode_Bekko_MatchesReference` (nuevo) | itera `testdata/bekko_reference_cases.json` con un `BPE` cargado de `testdata/bekko_tokenizer_config.json.gz` (mismo patrón que `loadTestBPE`, con `Scheme: MetaspaceScheme{}`) |
| `TestEncode_Bekko_ByteFallback` (nuevo) | `'ࡰ'` produce los símbolos `['▁','<0xE0>','<0xA1>','<0xB0>']` antes del merge (podés testear esto llamando `encodePretoken` directo, es un método no exportado del mismo paquete) y `Decode` reconstruye `' ࡰ'` exacto |
| `TestNew_RequiresScheme` (nuevo) | `New(Config{})` (sin `Scheme`) devuelve error, no pánico |
| Todos los existentes (`TestEncode_MatchesReference`, `TestIgnoreMerges_WholeWordInVocab`, `TestVocabLookup_BinarySearch`, `TestDecode_RoundTrip`, etc.) | siguen pasando, actualizados para pasar `Scheme: ByteLevelScheme{}` (y los tres ids de Granite) en cada `Config{}` que construyen |

`TestEncode_WrapsWithBosEos` cambia de comparar contra `179934`/`179938` (las constantes
borradas) a comparar contra el `BosTokenID`/`EosTokenID` que el propio test le pasó en su
`Config`.

## Checklist de aceptación

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
grep -rn "map\[" --include="*.go" . | grep -v _test.go              # → vacío
grep -rn '"errors"\|"regexp"' --include="*.go" . | grep -v _test.go   # → vacío
```

`"webtyp.com/fmt"` en `tokenizer.go` es el único import nuevo — no está en la lista de
prohibidos de `AGENTS.md` (esa lista prohíbe el `fmt` de la stdlib, no `webtyp.com/fmt`).
