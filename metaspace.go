package tokenizer

const metaspaceChar = '▁' // U+2581

// MetaspaceScheme is bekko-embedding-v1-a8m/a25m's pretokenizer + decoder: SentencePiece-style
// Metaspace (prepend_scheme=always, split=true) plus byte_fallback for out-of-vocab runes.
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
