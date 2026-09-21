package tokenizer

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
