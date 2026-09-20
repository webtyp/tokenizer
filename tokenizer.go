package tokenizer

import (
	"sort"
)

const (
	BosTokenID int32 = 179934 // <|startoftext|>
	PadTokenID int32 = 179935 // <|endoftext|>
	EosTokenID int32 = 179938 // <|return|>
)

// Config is the data a caller loads once per model.
type Config struct {
	// Vocab[id] is the byte-level token string for that id. Index IS the id.
	Vocab []string
	// Merges, in rank order (rank = index). Each entry is "left right".
	Merges []string
}

type vocabEntry struct {
	token string
	id    int32
}

type mergeEntry struct {
	left  string
	right string
	rank  int
}

// BPE is a ready-to-use tokenizer built from a Config.
type BPE struct {
	vocab        []string
	sortedVocab  []vocabEntry
	sortedMerges []mergeEntry
}

// New creates a new BPE tokenizer instance from a Config.
func New(cfg Config) (*BPE, error) {
	sortedVocab := make([]vocabEntry, len(cfg.Vocab))
	for i, v := range cfg.Vocab {
		sortedVocab[i] = vocabEntry{token: v, id: int32(i)}
	}
	sort.Slice(sortedVocab, func(i, j int) bool {
		return sortedVocab[i].token < sortedVocab[j].token
	})

	sortedMerges := make([]mergeEntry, 0, len(cfg.Merges))
	for rank, mergeStr := range cfg.Merges {
		left, right, ok := parseMergePair(mergeStr)
		if ok {
			sortedMerges = append(sortedMerges, mergeEntry{
				left:  left,
				right: right,
				rank:  rank,
			})
		}
	}
	sort.Slice(sortedMerges, func(i, j int) bool {
		if sortedMerges[i].left != sortedMerges[j].left {
			return sortedMerges[i].left < sortedMerges[j].left
		}
		return sortedMerges[i].right < sortedMerges[j].right
	})

	return &BPE{
		vocab:        cfg.Vocab,
		sortedVocab:  sortedVocab,
		sortedMerges: sortedMerges,
	}, nil
}

// Encode converts text into token IDs, prefixed with BOS and suffixed with EOS.
func (t *BPE) Encode(dst []int32, text string) []int32 {
	dst = append(dst, BosTokenID)

	pretokens := pretokenize(text)
	for _, pt := range pretokens {
		dst = t.encodePretoken(dst, pt)
	}

	dst = append(dst, EosTokenID)
	return dst
}

func (t *BPE) encodePretoken(dst []int32, pt string) []int32 {
	// ignore_merges: true optimization
	if id, ok := t.lookupVocab(pt); ok {
		return append(dst, id)
	}

	runes := []rune(pt)
	if len(runes) == 0 {
		return dst
	}

	symbols := make([]string, len(runes))
	for i, r := range runes {
		symbols[i] = string(r)
	}

	for len(symbols) > 1 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := t.lookupMergeRank(symbols[i], symbols[i+1]); ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
	}

	for _, sym := range symbols {
		if id, ok := t.lookupVocab(sym); ok {
			dst = append(dst, id)
		}
	}

	return dst
}

// Decode converts token IDs back to a text string, skipping special tokens.
func (t *BPE) Decode(ids []int32) string {
	var bytes []byte
	for _, id := range ids {
		if id == BosTokenID || id == EosTokenID || id == PadTokenID {
			continue
		}
		if int(id) < 0 || int(id) >= len(t.vocab) {
			continue
		}
		tokenStr := t.vocab[id]
		for _, r := range tokenStr {
			if b, ok := runeToByte(r); ok {
				bytes = append(bytes, b)
			} else {
				bytes = append(bytes, []byte(string(r))...)
			}
		}
	}
	return string(bytes)
}

// VocabSize returns the total number of items in the vocabulary.
func (t *BPE) VocabSize() int {
	return len(t.vocab)
}

func (t *BPE) lookupVocab(token string) (int32, bool) {
	idx := sort.Search(len(t.sortedVocab), func(i int) bool {
		return t.sortedVocab[i].token >= token
	})
	if idx < len(t.sortedVocab) && t.sortedVocab[idx].token == token {
		return t.sortedVocab[idx].id, true
	}
	return -1, false
}

func (t *BPE) lookupMergeRank(left, right string) (int, bool) {
	idx := sort.Search(len(t.sortedMerges), func(i int) bool {
		if t.sortedMerges[i].left != left {
			return t.sortedMerges[i].left >= left
		}
		return t.sortedMerges[i].right >= right
	})
	if idx < len(t.sortedMerges) && t.sortedMerges[idx].left == left && t.sortedMerges[idx].right == right {
		return t.sortedMerges[idx].rank, true
	}
	return -1, false
}

func parseMergePair(s string) (string, string, bool) {
	for i := 0; i < len(s); i++ {
		if s[i] == ' ' {
			return s[:i], s[i+1:], true
		}
	}
	return "", "", false
}
