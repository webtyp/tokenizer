package tokenizer

import "sort"

// composeLatin performs NFC normalization restricted to Latin letters using the nfcLatin table.
// If text contains no combining mark in U+0300–U+036F, it returns text unchanged without allocations.
func composeLatin(text string) string {
	hasCombining := false
	for _, r := range text {
		if r >= 0x0300 && r <= 0x036F {
			hasCombining = true
			break
		}
	}
	if !hasCombining {
		return text
	}

	runes := []rune(text)
	out := make([]rune, 0, len(runes))

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if len(out) > 0 && r >= 0x0300 && r <= 0x036F {
			last := out[len(out)-1]
			comp, ok := lookupNFCLatin(last, r)
			if ok {
				out[len(out)-1] = comp
				// Keep attempting to compose stacked marks with the newly composed base
				for i+1 < len(runes) && runes[i+1] >= 0x0300 && runes[i+1] <= 0x036F {
					nextMark := runes[i+1]
					nextComp, ok2 := lookupNFCLatin(out[len(out)-1], nextMark)
					if ok2 {
						out[len(out)-1] = nextComp
						i++
					} else {
						break
					}
				}
				continue
			}
		}
		out = append(out, r)
	}

	return string(out)
}

func lookupNFCLatin(base, mark rune) (rune, bool) {
	idx := sort.Search(len(nfcLatin), func(i int) bool {
		if nfcLatin[i][0] != base {
			return nfcLatin[i][0] >= base
		}
		return nfcLatin[i][1] >= mark
	})
	if idx < len(nfcLatin) && nfcLatin[idx][0] == base && nfcLatin[idx][1] == mark {
		return nfcLatin[idx][2], true
	}
	return 0, false
}
