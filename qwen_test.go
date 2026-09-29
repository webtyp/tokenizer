package tokenizer

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type qwenTestCase struct {
	Text       string   `json:"text"`
	Normalized string   `json:"normalized"`
	Pretokens  []string `json:"pretokens"`
	IDs        []int32  `json:"ids"`
}

type qwenTokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

func loadQwenCases(t *testing.T) []qwenTestCase {
	data, err := os.ReadFile("testdata/qwen35_pretokenize.json")
	if err != nil {
		t.Fatalf("failed to read testdata/qwen35_pretokenize.json: %v", err)
	}
	var cases []qwenTestCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal qwen35_pretokenize.json: %v", err)
	}
	return cases
}

func loadQwenBPE(t *testing.T) *BPE {
	f, err := os.Open("testdata/qwen35_tokenizer.json.gz")
	if err != nil {
		t.Fatalf("failed to open testdata/qwen35_tokenizer.json.gz: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("failed to create gzip reader: %v", err)
	}
	defer gz.Close()

	var raw qwenTokenizerJSON
	if err := json.NewDecoder(gz).Decode(&raw); err != nil {
		t.Fatalf("failed to decode JSON tokenizer config: %v", err)
	}

	maxID := -1
	for _, id := range raw.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, at := range raw.AddedTokens {
		if at.ID > maxID {
			maxID = at.ID
		}
	}

	vocab := make([]string, maxID+1)
	for tok, id := range raw.Model.Vocab {
		vocab[id] = tok
	}
	for _, at := range raw.AddedTokens {
		vocab[at.ID] = at.Content
	}

	bpe, err := New(Config{
		Vocab:      vocab,
		Merges:     raw.Model.Merges,
		Scheme:     QwenScheme{},
		BosTokenID: -1,
		EosTokenID: -1,
		PadTokenID: -1,
	})
	if err != nil {
		t.Fatalf("failed to create Qwen BPE: %v", err)
	}
	return bpe
}

func TestQwen_Normalize(t *testing.T) {
	cases := loadQwenCases(t)
	for i, tc := range cases {
		got := composeLatin(tc.Text)
		if got != tc.Normalized {
			t.Errorf("case %d (%q):\n got:  %q\n want: %q", i, tc.Text, got, tc.Normalized)
		}
	}
}

func TestQwen_Pretokenize(t *testing.T) {
	scheme := QwenScheme{}
	cases := loadQwenCases(t)
	for i, tc := range cases {
		got := scheme.Pretokenize(tc.Text)
		want := tc.Pretokens
		if len(got) == 0 && len(want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.Text, got, want)
		}
	}
}

func TestQwen_EncodeMatchesReference(t *testing.T) {
	bpe := loadQwenBPE(t)
	cases := loadQwenCases(t)
	for i, tc := range cases {
		// Case 15 contains added tokens (<tool_call>, </tool_call>) which Hugging Face's
		// tokenizer.encode replaces before pretokenization. Special/added token matching
		// is performed by webtyp/qwen rather than Scheme.Pretokenize.
		if i == 15 {
			continue
		}
		got := bpe.EncodeOrdinary(nil, tc.Text)
		want := tc.IDs
		if len(got) == 0 && len(want) == 0 {
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.Text, got, want)
		}
	}
}

func TestQwen_DecodeRoundTrip(t *testing.T) {
	bpe := loadQwenBPE(t)
	cases := loadQwenCases(t)
	for i, tc := range cases {
		ids := bpe.EncodeOrdinary(nil, tc.Text)
		got := bpe.Decode(ids)
		if got != tc.Normalized {
			t.Errorf("case %d (%q):\n got:  %q\n want: %q", i, tc.Text, got, tc.Normalized)
		}
	}
}

func TestEncode_IsBosOrdinaryEos(t *testing.T) {
	bpe := loadTestBPE(t)
	text := "Hola mundo"
	full := bpe.Encode(nil, text)
	ordinary := bpe.EncodeOrdinary(nil, text)

	want := make([]int32, 0, len(ordinary)+2)
	want = append(want, bpe.bosTokenID)
	want = append(want, ordinary...)
	want = append(want, bpe.eosTokenID)

	if !reflect.DeepEqual(full, want) {
		t.Errorf("Encode != [bos] + EncodeOrdinary + [eos]:\n got:  %v\n want: %v", full, want)
	}
}

func TestComposeLatin_NoMarksNoAlloc(t *testing.T) {
	allocs := testing.AllocsPerRun(100, func() {
		_ = composeLatin("hola mundo")
	})
	if allocs != 0 {
		t.Errorf("expected 0 allocs for text without marks, got %f", allocs)
	}
}
