package tokenizer

// Stream turns the bytes of generated tokens into text that never ends in half a character. A
// byte-level token (Qwen, LFM2) can end in the middle of a multi-byte UTF-8 character, and
// showing that half prints garbage; Stream holds such a tail back until the character completes.
type Stream struct {
	text  []byte
	shown int
}

// Write appends one token's bytes (Scheme.DecodeToken) and returns the text that became
// complete, possibly "".
func (s *Stream) Write(b []byte) string {
	s.text = append(s.text, b...)
	end := completeUTF8(s.text)
	if end <= s.shown {
		return ""
	}
	out := string(s.text[s.shown:end])
	s.shown = end
	return out
}

// Flush returns whatever Write held back: at the end of generation, an incomplete tail is shown
// as it is.
func (s *Stream) Flush() string {
	out := string(s.text[s.shown:])
	s.shown = len(s.text)
	return out
}

// Text returns all the bytes written so far, held-back tail included. Every Write result joined
// with Flush equals Text.
func (s *Stream) Text() string { return string(s.text) }

// completeUTF8 returns how many leading bytes of b can be shown now: all of b, except a trailing
// lead byte (and its continuation bytes) of a multi-byte character that still misses bytes.
func completeUTF8(b []byte) int {
	for back := 1; back <= 3 && back <= len(b); back++ {
		c := b[len(b)-back]
		if c&0xC0 == 0x80 { // continuation byte: the lead is further back
			continue
		}
		need := 1
		switch {
		case c&0xE0 == 0xC0:
			need = 2
		case c&0xF0 == 0xE0:
			need = 3
		case c&0xF8 == 0xF0:
			need = 4
		}
		if need > back {
			return len(b) - back
		}
		return len(b)
	}
	return len(b)
}
