package tokenizer

import "unicode"

// Lfm2Scheme implements Scheme for LFM2 models (LiquidAI LFM2.5): no normalization, a GPT-4-style
// split that keeps up to three digits together, then byte-level.
type Lfm2Scheme struct{}

func (Lfm2Scheme) Pretokenize(text string) []string {
	if len(text) == 0 {
		return nil
	}

	runes := []rune(text)
	n := len(runes)
	var tokens []string
	i := 0

	for i < n {
		length := matchLfm2NextToken(runes, i, n)
		if length <= 0 {
			length = 1
		}
		chunk := string(runes[i : i+length])
		tokens = append(tokens, encodeBytes([]byte(chunk)))
		i += length
	}

	return tokens
}

func (Lfm2Scheme) ByteFallbackSymbol(b byte) (string, bool) {
	return "", false
}

func (Lfm2Scheme) DecodeToken(dst []byte, tok string) []byte {
	return ByteLevelScheme{}.DecodeToken(dst, tok)
}

func (Lfm2Scheme) IgnoreMerges() bool {
	return false
}

func matchLfm2NextToken(runes []rune, i, n int) int {
	// Alt 1: Contraction
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if l := matchQwenContraction(runes, i, n); l > 0 {
		return l
	}

	// Alt 2: Word
	// [^\r\n\p{L}\p{N}]?\p{L}+
	if l := matchLfm2Word(runes, i, n); l > 0 {
		return l
	}

	// Alt 3: Number
	// \p{N}{1,3}
	if l := matchLfm2Number(runes, i, n); l > 0 {
		return l
	}

	// Alt 4: Punctuation
	//  ?[^\s\p{L}\p{N}]+[\r\n]*
	if l := matchLfm2Punct(runes, i, n); l > 0 {
		return l
	}

	// Alt 5: Newlines
	// \s*[\r\n]+
	if l := matchQwenNewlines(runes, i, n); l > 0 {
		return l
	}

	// Alt 6 & 7: Trailing Whitespace / Whitespace
	// \s+(?!\S) | \s+
	if l := matchQwenSpace(runes, i, n); l > 0 {
		return l
	}

	return 1
}

func matchLfm2Word(runes []rune, i, n int) int {
	curr := i
	if curr < n {
		r := runes[curr]
		if !isCRLF(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
			curr++
		}
	}

	lCount := 0
	for curr+lCount < n && unicode.IsLetter(runes[curr+lCount]) {
		lCount++
	}

	if lCount == 0 {
		return 0
	}

	return (curr - i) + lCount
}

func matchLfm2Number(runes []rune, i, n int) int {
	count := 0
	for i+count < n && count < 3 && unicode.IsNumber(runes[i+count]) {
		count++
	}
	return count
}

func matchLfm2Punct(runes []rune, i, n int) int {
	curr := i
	if curr < n && runes[curr] == ' ' {
		curr++
	}

	startNonSym := curr
	for curr < n && !unicode.IsSpace(runes[curr]) && !unicode.IsLetter(runes[curr]) && !unicode.IsNumber(runes[curr]) {
		curr++
	}

	if curr == startNonSym {
		return 0
	}

	for curr < n && isCRLF(runes[curr]) {
		curr++
	}

	return curr - i
}
