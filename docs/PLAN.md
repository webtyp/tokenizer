---
PLAN: "feat: webtyp/tokenizer — byte-level BPE for granite-embedding-97m-multilingual-r2"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 6379221482996001613
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md
>
> **Este plan reemplaza por completo a la versión anterior** (que vivía en
> `agent/docs/plans/tokenizer.md` y asumía WordPiece). Esa asunción era incorrecta para el
> modelo que el proyecto eligió: `granite-embedding-97m-multilingual-r2` usa **BPE a nivel de
> byte**, no WordPiece. Todo lo de abajo está verificado contra el `tokenizer.json` real del
> modelo (HuggingFace, `ibm-granite/granite-embedding-97m-multilingual-r2`), no inferido.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés.

# Plan — `webtyp/tokenizer`

## Responsabilidad única

Entra texto, salen ids de tokens. Nada más. Sin embeddings, sin pesos de modelo, sin
almacenamiento, cero dependencias fuera de `webtyp.com/fmt`. Compila bajo TinyGo para WASM —
es la pieza que corre en el navegador para tokenizar la consulta, y en el backend para
tokenizar documentos.

## Design gate

**1. Prior art.** 🤗 `tokenizers` (Rust, la implementación de referencia de este modelo
exacto — es la que generó `tokenizer.json`); OpenAI `tiktoken` (Python/Rust, el mismo estilo
de BPE a nivel de byte con `ignore_merges`, que es de donde viene este diseño); `sentencepiece`
(Google, Unigram — **no aplica acá**, es un algoritmo distinto). Este plan porta el
comportamiento exacto de `tokenizers`, no reinventa BPE: la especificación es el
`tokenizer.json` real del modelo, no un paper.

**2. Novice-name test.** `tokenizer.New(vocab, merges) (*BPE, error)` — nombre y forma
calcados de la familia `tokenizers`/`tiktoken`; `Encode`/`Decode` son los verbos que ya usa
todo el ecosistema Python/Rust de NLP. Nada nuevo que aprender.

**3. Complexity ledger.**
```
Conceptos nuevos                          +2 (BPE merge, pretokenización por regex-equivalente)
Formas de tokenizar en el repo            1  (esta librería; nada más la reimplementa)
Dependencia de `map[string]int32`         −1 (el plan anterior la tenía; este la elimina —
                                              ver la sección de layout de datos abajo)
```

**4. Dónde vive.** Repo aparte de `embed`, como ya decidía el plan anterior: tokenizar es
idéntico para las dos fases del embedder (tabla estática y transformer) y es donde viven los
bugs silenciosos más caros — merece su propia suite de tests versionada aparte.

**5. Qué borra.** El plan anterior completo (WordPiece, `lowercase`/`strip_accents`,
`map[string]int32`). No queda código de ese diseño en este repo — nunca se escribió, así que
no hay nada que limpiar, pero si encontrás algún resto asumiendo WordPiece, es el defecto a
corregir, no una opción a mantener detrás de un flag.

## Por qué tiene que ser exacto

Un tokenizador que discrepa del usado para entrenar el modelo produce embeddings
*plausibles pero incorrectos*: sin crash, apenas una recuperación degradada en silencio que
parece un modelo malo. Cada test de la sección Tests fija comportamiento contra la
implementación de referencia (🤗 `tokenizers`, corrida en Python), no contra la lógica que
este repo termine escribiendo — si el test y el código coinciden pero los dos están mal, el
test no vale nada.

## El algoritmo real, verificado desde `tokenizer.json`

```
"normalizer": null                          ← sin lowercase, sin strip de acentos, sin NFKC
"pre_tokenizer": Sequence[ Split(regex, "Isolated"), ByteLevel(add_prefix_space) ]
"model": { "type": "BPE", "ignore_merges": true, "byte_fallback": false, "unk_token": null }
"decoder": ByteLevel
"post_processor": TemplateProcessing — envuelve cada secuencia como
    <|startoftext|>  ...ids...  <|return|>
```

Cuatro pasos, en orden:

### 1. Pretokenización por regex

El regex real (cortalo del `tokenizer.json`, no lo reescribas de memoria):

```
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
|\p{N}{1,3}
| ?[^\s\p{L}\p{N}]+[\r\n/]*
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

**Bloqueante conocido, resolvelo así:** el último grupo de alternativas usa
`\s+(?!\S)` — un *negative lookahead*. El paquete `regexp` de Go (motor RE2) **no soporta
lookahead ni lookbehind**; este patrón no compila tal cual con `regexp.Compile`. No es un
detalle menor y no tiene solución con una librería de regex de terceros (nada de cgo, nada de
Oniguruma — este código compila a WASM bajo TinyGo).

La solución es escribir un scanner a mano que reproduzca el mismo resultado, alternativa por
alternativa, usando las tablas de `unicode` de la stdlib de Go (`unicode.IsLetter`,
`unicode.Is(unicode.Lu, r)`, `unicode.Is(unicode.M, r)`, etc. — todas existen y **sí**
compilan bajo TinyGo; verificalo con `tinygo build`, es parte del checklist). El único
fragmento genuinamente difícil es `\s+(?!\S)`: significa "una corrida de espacios que NO
sigue inmediatamente un no-espacio" — en la práctica, dado que las alternativas están en
orden de prioridad y las anteriores ya consumen espacio-más-token, esta rama solo dispara al
**final de la cadena** (una corrida de espacios sin nada después). Reproducila así: si una
corrida de espacios llega hasta el final del input, consumila entera; si no, dejá el último
carácter de espacio sin consumir (para que empiece el siguiente token, vía la rama genérica
`\s+`) y consumí el resto. Esto es exactamente lo que produce el lookahead, sin necesitar uno.

**No hay forma de saltarse esto:** confirmá el resultado exacto contra 🤗 `tokenizers` con el
fixture de la sección Tests, no contra tu propia lectura del regex.

### 2. Codificación byte-level

Cada pretoken se re-codifica byte por byte: cada uno de los 256 valores de byte crudo se
mapea a un carácter Unicode imprimible fijo (la tabla estándar de GPT-2 — 188 bytes ya
imprimibles se mapean a sí mismos; los 68 restantes al rango `U+0100`–`U+0143`). Es una tabla
de 256 entradas, fija, va como constante — no hace falta calcularla en runtime ni un `map`.

`add_prefix_space: true` (verificalo en el `tokenizer.json` real) significa: si el texto no
empieza con un espacio, prependé uno antes de pretokenizar. Esto es lo que hace que "hola" al
inicio de una oración y "hola" después de otra palabra tokenicen igual (con el espacio
prefijado codificado dentro del primer symbol byte-level).

### 3. Merge BPE, con `ignore_merges: true`

Para cada pretoken ya codificado en símbolos byte-level:

```
si el pretoken completo (como string) existe directo en vocab:
    emitilo como un solo id — NO corras el loop de merge
si no:
    símbolos = [cada carácter byte-level del pretoken, uno por uno]
    repetir:
        de todos los pares adyacentes en símbolos, encontrá el de MENOR rank en merges
        si ninguno tiene rank (no hay más merges aplicables): cortar
        fusioná ese par en símbolos
    mapear cada símbolo final a su id de vocab
```

`ignore_merges: true` es la optimización que usan `tiktoken` y los tokenizers estilo GPT-4/o200k:
evita miles de pasos de merge para palabras comunes que ya están enteras en el vocabulario de
180k. Sin este atajo el resultado final es el mismo, pero mucho más lento — impleméntalo, no
es opcional para el rendimiento en un query de navegador.

### 4. Envoltorio de secuencia

`<|startoftext|>` (id 179934) al principio, `<|return|>` (id 179938) al final — la misma
constante para BOS/CLS y para EOS/SEP (verificado en `config.json`:
`bos_token_id == cls_token_id`, `eos_token_id == sep_token_id`). Esto lo hace `Encode`, no el
llamador.

**Fuera de alcance de este plan:** padding (`<|endoftext|>`, id 179935) y `[MASK]` — no hacen
falta para embeber una consulta ni un documento completo; si el backend necesita batching con
padding más adelante, es un plan aparte.

## Layout de datos — sin `map[K]V`

```go
// Config is the data a caller loads once per model (from weights.Artifact.Tokenizer.Vocab
// plus the .merges file webtyp/weightsc produces — see its docs/PLAN.md).
type Config struct {
	// Vocab[id] is the byte-level token string for that id. Index IS the id — no lookup
	// structure needed for id → string.
	Vocab []string
	// Merges, in rank order (rank = index). Each entry is "left right", the two symbols
	// a merge rule combines, matching tokenizer.json's merges list verbatim.
	Merges []string
}

// BPE is a ready-to-use tokenizer built from a Config.
type BPE struct { /* unexported: sorted lookup tables built once in New */ }

func New(cfg Config) (*BPE, error)

func (t *BPE) Encode(dst []int32, text string) []int32
func (t *BPE) Decode(ids []int32) string
func (t *BPE) VocabSize() int
```

`New` construye, una sola vez, dos slices **ordenados** (vocab por string, merges por el par
`(left, right)`) y busca en ellos con `sort.Search` (búsqueda binaria, `O(log n)`, ninguna
importación prohibida). Con 180 000 entradas eso son ~18 comparaciones por búsqueda — nada
comparado con el I/O que ya paga cargar el artifact. Esto es la regla del proyecto
(`fmt.KeyValue` o slice+scan en vez de `map[K]V`) aplicada al caso donde un scan lineal sí
sería demasiado lento: ordenar una vez y buscar en binario, en vez de un `map`.

## Tests

Fixture generado UNA VEZ desde la implementación de referencia real (Python, `tokenizers` +
`AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-97m-multilingual-r2")`),
checkeado al repo como JSON (`testdata/reference_cases.json`): una lista de
`{"text": ..., "ids": [...]}`. Incluí, como mínimo: texto en español con acentos y mayúsculas,
una oración con números, puntuación pegada a palabras, espacios múltiples, y una cadena vacía.
**No inventes los ids esperados a mano** — vienen de correr el tokenizer real una vez.

| Test | Verifica |
|---|---|
| `TestEncode_MatchesReference` | itera `testdata/reference_cases.json`, compara ids exactos |
| `TestEncode_WrapsWithBosEos` | el primer id es 179934, el último 179938, para texto no vacío |
| `TestEncode_EmptyString` | no pánica; produce al menos BOS+EOS |
| `TestPretokenize_TrailingWhitespace` | el caso `\s+(?!\S)` puntual: una cadena que termina en espacios round-tripea igual que la referencia |
| `TestIgnoreMerges_WholeWordInVocab` | una palabra completa presente en vocab se tokeniza como un solo id, sin correr merges |
| `TestVocabLookup_BinarySearch` | `sort.Search` sobre el vocab ordenado encuentra cada entrada real y falla (retorna "no encontrado", no pánico) para una que no existe |
| `TestDecode_RoundTrip` | `Decode(Encode(x)) == x` para texto ASCII simple (el byte-level lo garantiza) |

## Checklist de aceptación

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
tinygo build -target wasm -o /dev/null .
grep -rn "map\[" --include="*.go" . | grep -v _test.go              # → vacío
grep -rn '"errors"\|"fmt"\|"regexp"' --include="*.go" . | grep -v _test.go   # → vacío
```

`regexp` está explícitamente prohibido en el build final: aunque compilara bajo TinyGo (no
verificado, y el lookahead de arriba lo rompe de todos modos), el motor de regex completo es
peso que un scanner a mano no paga. Usalo, si querés, para prototipar en un test que no se
compila al binario final — nunca en el código de producción de este repo.

## La comparación que este plan existe para producir

Corré el checklist completo bajo **los tres** targets y registrá los tres números/resultados
en el README, lado a lado — `gotest` (Go stdlib, target `js/wasm` vía `wasmbrowsertest`),
`gotest -tinygo` (TinyGo), y nativo (`go test .` sin flags). Si algo compila o pasa bajo un
target y no bajo otro, documentalo explícitamente con el mensaje de error real — es
exactamente el dato que el índice maestro está pidiendo de esta tanda de planes.
