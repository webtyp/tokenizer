package tokenizer

import "unicode"

// QwenScheme implements Scheme for Qwen3.5 models.
// Normalization uses composeLatin (NFC restricted to Latin script).
// Pretokenization splits according to Qwen3.5's regex alternatives in order.
type QwenScheme struct{}

func (QwenScheme) Pretokenize(text string) []string {
	normalized := composeLatin(text)
	if len(normalized) == 0 {
		return nil
	}

	runes := []rune(normalized)
	n := len(runes)
	var tokens []string
	i := 0

	for i < n {
		length := matchQwenNextToken(runes, i, n)
		if length <= 0 {
			length = 1
		}
		chunk := string(runes[i : i+length])
		tokens = append(tokens, encodeBytes([]byte(chunk)))
		i += length
	}

	return tokens
}

func (QwenScheme) ByteFallbackSymbol(b byte) (string, bool) {
	return "", false
}

func (QwenScheme) DecodeToken(dst []byte, tok string) []byte {
	return ByteLevelScheme{}.DecodeToken(dst, tok)
}

func matchQwenNextToken(runes []rune, i, n int) int {
	// Alt 1: Contraction
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if l := matchQwenContraction(runes, i, n); l > 0 {
		return l
	}

	// Alt 2: Word
	// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
	if l := matchQwenWord(runes, i, n); l > 0 {
		return l
	}

	// Alt 3: Number
	// \p{N}
	if l := matchQwenNumber(runes, i, n); l > 0 {
		return l
	}

	// Alt 4: Punctuation
	//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
	if l := matchQwenPunct(runes, i, n); l > 0 {
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

func matchQwenContraction(runes []rune, i, n int) int {
	if i >= n || runes[i] != '\'' {
		return 0
	}
	rem := n - (i + 1)
	if rem < 1 {
		return 0
	}

	r1 := toLower(runes[i+1])
	if r1 == 's' || r1 == 't' || r1 == 'm' || r1 == 'd' {
		return 2
	}

	if rem >= 2 {
		r2 := toLower(runes[i+2])
		if (r1 == 'r' && r2 == 'e') || (r1 == 'v' && r2 == 'e') || (r1 == 'l' && r2 == 'l') {
			return 3
		}
	}

	return 0
}

func isMark(r rune) bool {
	return unicode.Is(unicode.M, r)
}

func isQwenLetterOrMark(r rune) bool {
	return unicode.IsLetter(r) || isMark(r)
}

func matchQwenWord(runes []rune, i, n int) int {
	curr := i
	if curr < n {
		r := runes[curr]
		if !isCRLF(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r) {
			curr++
		}
	}

	lmCount := 0
	for curr+lmCount < n && isQwenLetterOrMark(runes[curr+lmCount]) {
		lmCount++
	}

	if lmCount == 0 {
		return 0
	}

	return (curr - i) + lmCount
}

func matchQwenNumber(runes []rune, i, n int) int {
	if i < n && unicode.IsNumber(runes[i]) {
		return 1
	}
	return 0
}

func matchQwenPunct(runes []rune, i, n int) int {
	curr := i
	if curr < n && runes[curr] == ' ' {
		curr++
	}

	startNonSym := curr
	for curr < n && !unicode.IsSpace(runes[curr]) && !unicode.IsLetter(runes[curr]) && !isMark(runes[curr]) && !unicode.IsNumber(runes[curr]) {
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

func matchQwenNewlines(runes []rune, i, n int) int {
	curr := i
	lastCRLFEnd := -1

	for curr < n && unicode.IsSpace(runes[curr]) {
		if isCRLF(runes[curr]) {
			curr++
			for curr < n && isCRLF(runes[curr]) {
				curr++
			}
			lastCRLFEnd = curr
		} else {
			curr++
		}
	}

	if lastCRLFEnd == -1 {
		return 0
	}

	return lastCRLFEnd - i
}

func matchQwenSpace(runes []rune, i, n int) int {
	wsCount := 0
	for i+wsCount < n && unicode.IsSpace(runes[i+wsCount]) {
		wsCount++
	}

	if wsCount == 0 {
		return 0
	}

	// Alt 6: \s+(?!\S)
	// If at end of string, \s+(?!\S) matches full wsCount
	if i+wsCount == n {
		return wsCount
	}

	// If followed by non-space (\S):
	// If wsCount > 1, \s+(?!\S) matches wsCount - 1
	if wsCount > 1 {
		return wsCount - 1
	}

	// If wsCount == 1 and followed by \S, \s+(?!\S) fails, Alt 7 \s+ matches 1 space
	return 1
}
