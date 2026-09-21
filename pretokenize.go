package tokenizer

import (
	"unicode"
)

// pretokenize splits text according to the model's pretokenizer rules:
// 1. Split text into pretoken chunks using the reference regex alternatives scanner.
// 2. Convert each chunk bytes into byte-level representation string using GPT-2 byte mapping.
func pretokenize(text string) []string {
	if len(text) == 0 {
		return nil
	}

	runes := []rune(text)
	n := len(runes)
	var tokens []string
	i := 0

	for i < n {
		length := matchNextToken(runes, i, n)
		if length <= 0 {
			length = 1
		}
		chunk := string(runes[i : i+length])
		tokens = append(tokens, encodeBytes([]byte(chunk)))
		i += length
	}

	return tokens
}

func matchNextToken(runes []rune, i, n int) int {
	// Alt 1 & Alt 2:
	// Alt 1: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
	// Alt 2: [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
	if l := matchAlt12(runes, i, n); l > 0 {
		return l
	}

	// Alt 3: \p{N}{1,3}
	if l := matchAlt3(runes, i, n); l > 0 {
		return l
	}

	// Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
	if l := matchAlt4(runes, i, n); l > 0 {
		return l
	}

	// Alt 5: \s*[\r\n]+
	if l := matchAlt5(runes, i, n); l > 0 {
		return l
	}

	// Alt 6 & 7: \s+(?!\S) | \s+
	if l := matchAlt67(runes, i, n); l > 0 {
		return l
	}

	return 1
}

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

func matchAlt3(runes []rune, i, n int) int {
	count := 0
	for i+count < n && count < 3 && isNumber(runes[i+count]) {
		count++
	}
	return count
}

func matchAlt4(runes []rune, i, n int) int {
	curr := i
	if curr < n && runes[curr] == ' ' {
		curr++
	}

	nonSymStart := curr
	for curr < n && !isSpace(runes[curr]) && !isLetter(runes[curr]) && !isNumber(runes[curr]) {
		curr++
	}

	if curr == nonSymStart {
		return 0
	}

	for curr < n && (runes[curr] == '\r' || runes[curr] == '\n' || runes[curr] == '/') {
		curr++
	}

	return curr - i
}

func matchAlt5(runes []rune, i, n int) int {
	curr := i
	lastCRLFEnd := -1

	for curr < n && isSpace(runes[curr]) {
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

func matchAlt67(runes []rune, i, n int) int {
	wsCount := 0
	for i+wsCount < n && isSpace(runes[i+wsCount]) {
		wsCount++
	}

	if wsCount == 0 {
		return 0
	}

	// If whitespace run reaches end of text, consume all (\s+(?!\S))
	if i+wsCount == n {
		return wsCount
	}

	// If whitespace run is followed by non-whitespace (\S):
	// \s+(?!\S) matches wsCount-1 spaces if wsCount > 1.
	if wsCount > 1 {
		return wsCount - 1
	}

	// If wsCount == 1, \s+(?!\S) fails, \s+ matches 1 space.
	return 1
}

func matchContraction(runes []rune, i, n int) int {
	if i >= n {
		return 0
	}

	if runes[i] != '\'' {
		return 0
	}

	rem := n - (i + 1)
	if rem < 1 {
		return 0
	}

	r1 := toLower(runes[i+1])

	// 1-char suffixes: 's, 't, 'm, 'd
	if r1 == 's' || r1 == 't' || r1 == 'm' || r1 == 'd' {
		return 2
	}

	if rem >= 2 {
		r2 := toLower(runes[i+2])
		// 2-char suffixes: 're, 've, 'll
		if (r1 == 'r' && r2 == 'e') || (r1 == 'v' && r2 == 'e') || (r1 == 'l' && r2 == 'l') {
			return 3
		}
	}

	return 0
}

func isCRLF(r rune) bool {
	return r == '\r' || r == '\n'
}

func isLetter(r rune) bool {
	return unicode.IsLetter(r)
}

func isNumber(r rune) bool {
	return unicode.IsNumber(r)
}

func isSpace(r rune) bool {
	return unicode.IsSpace(r)
}

func isUpperLike(r rune) bool {
	return unicode.Is(unicode.Lu, r) || unicode.Is(unicode.Lt, r) || unicode.Is(unicode.Lm, r) || unicode.Is(unicode.Lo, r) || unicode.Is(unicode.M, r)
}

func isLowerLike(r rune) bool {
	return unicode.Is(unicode.Ll, r) || unicode.Is(unicode.Lm, r) || unicode.Is(unicode.Lo, r) || unicode.Is(unicode.M, r)
}

func toLower(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + ('a' - 'A')
	}
	return r
}
