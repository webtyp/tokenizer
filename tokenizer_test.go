package tokenizer

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type refCase struct {
	Text string  `json:"text"`
	IDs  []int32 `json:"ids"`
}

type loadedConfig struct {
	Vocab  []string `json:"vocab"`
	Merges []string `json:"merges"`
}

func loadTestBPE(t *testing.T) *BPE {
	f, err := os.Open("testdata/tokenizer_config.json.gz")
	if err != nil {
		t.Fatalf("failed to open testdata/tokenizer_config.json.gz: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("failed to create gzip reader: %v", err)
	}
	defer gz.Close()

	var cfg loadedConfig
	if err := json.NewDecoder(gz).Decode(&cfg); err != nil {
		t.Fatalf("failed to decode JSON config: %v", err)
	}

	bpe, err := New(Config{
		Vocab:      cfg.Vocab,
		Merges:     cfg.Merges,
		Scheme:     ByteLevelScheme{},
		BosTokenID: 179934,
		PadTokenID: 179935,
		EosTokenID: 179938,
	})
	if err != nil {
		t.Fatalf("failed to create BPE: %v", err)
	}
	return bpe
}

func loadBekkoBPE(t *testing.T) *BPE {
	f, err := os.Open("testdata/bekko_tokenizer_config.json.gz")
	if err != nil {
		t.Fatalf("failed to open testdata/bekko_tokenizer_config.json.gz: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatalf("failed to create gzip reader: %v", err)
	}
	defer gz.Close()

	var cfg loadedConfig
	if err := json.NewDecoder(gz).Decode(&cfg); err != nil {
		t.Fatalf("failed to decode JSON config: %v", err)
	}

	bpe, err := New(Config{
		Vocab:      cfg.Vocab,
		Merges:     cfg.Merges,
		Scheme:     MetaspaceScheme{},
		BosTokenID: 2,
		PadTokenID: 0,
		EosTokenID: 1,
	})
	if err != nil {
		t.Fatalf("failed to create BPE: %v", err)
	}
	return bpe
}

func TestNew_RequiresScheme(t *testing.T) {
	_, err := New(Config{})
	if err == nil {
		t.Errorf("expected error when Config.Scheme is nil, got nil")
	}
}

func TestEncode_MatchesReference(t *testing.T) {
	bpe := loadTestBPE(t)

	data, err := os.ReadFile("testdata/reference_cases.json")
	if err != nil {
		t.Fatalf("failed to read testdata/reference_cases.json: %v", err)
	}

	var cases []refCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal reference cases: %v", err)
	}

	for i, tc := range cases {
		got := bpe.Encode(nil, tc.Text)
		if !reflect.DeepEqual(got, tc.IDs) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.Text, got, tc.IDs)
		}
	}
}

func TestEncode_WrapsWithBosEos(t *testing.T) {
	bpe := loadTestBPE(t)
	ids := bpe.Encode(nil, "Hola")
	if len(ids) < 2 {
		t.Fatalf("expected at least 2 token IDs, got %d", len(ids))
	}
	if ids[0] != 179934 {
		t.Errorf("expected first ID to be BOS (179934), got %d", ids[0])
	}
	if ids[len(ids)-1] != 179938 {
		t.Errorf("expected last ID to be EOS (179938), got %d", ids[len(ids)-1])
	}
}

func TestEncode_EmptyString(t *testing.T) {
	bpe := loadTestBPE(t)
	ids := bpe.Encode(nil, "")
	want := []int32{179934, 179938}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("got %v, want %v", ids, want)
	}
}

func TestIgnoreMerges_WholeWordInVocab(t *testing.T) {
	bpe := loadTestBPE(t)

	// "Hola" is in vocab (ID 47970).
	// With ignore_merges: true, pretoken "Hola" is encoded as a single token without merge iteration.
	ids := bpe.Encode(nil, "Hola")
	want := []int32{179934, 47970, 179938}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("got %v, want %v", ids, want)
	}
}

func TestVocabLookup_BinarySearch(t *testing.T) {
	bpe := loadTestBPE(t)

	id, ok := bpe.lookupVocab("Hola")
	if !ok || id != 47970 {
		t.Errorf("expected id 47970 for 'Hola', got id=%d ok=%v", id, ok)
	}

	_, ok = bpe.lookupVocab("NON_EXISTENT_TOKEN_XYZ_12345")
	if ok {
		t.Errorf("expected lookup for non-existent token to fail")
	}
}

func TestDecode_RoundTrip(t *testing.T) {
	bpe := loadTestBPE(t)
	input := "Hola mundo ASCII 123"
	ids := bpe.Encode(nil, input)
	decoded := bpe.Decode(ids)
	if decoded != input {
		t.Errorf("Decode(Encode(%q)) = %q, want %q", input, decoded, input)
	}
}

func TestVocabSize(t *testing.T) {
	bpe := loadTestBPE(t)
	if bpe.VocabSize() != 180000 {
		t.Errorf("expected VocabSize to be 180000, got %d", bpe.VocabSize())
	}
}

func TestMetaspacePretokenize(t *testing.T) {
	scheme := MetaspaceScheme{}
	tests := []struct {
		input string
		want  []string
	}{
		{"hola mundo", []string{"▁hola", "▁mundo"}},
		{"Hola Mundo", []string{"▁Hola", "▁Mundo"}},
		{"  espacios   múltiples  ", []string{"▁", "▁espacios", "▁", "▁", "▁múltiples", "▁", "▁"}},
		{"café niño mañana", []string{"▁café", "▁niño", "▁mañana"}},
		{"你好世界", []string{"▁你好世界"}},
		{"你A", []string{"▁你A"}},
		{"a", []string{"▁a"}},
		{" a", []string{"▁a"}},
		{"a ", []string{"▁a", "▁"}},
		{"😀 emoji test", []string{"▁😀", "▁emoji", "▁test"}},
		{"", nil},
	}

	for i, tc := range tests {
		got := scheme.Pretokenize(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.input, got, tc.want)
		}
	}
}

func TestEncode_Bekko_MatchesReference(t *testing.T) {
	bpe := loadBekkoBPE(t)

	data, err := os.ReadFile("testdata/bekko_reference_cases.json")
	if err != nil {
		t.Fatalf("failed to read testdata/bekko_reference_cases.json: %v", err)
	}

	var cases []refCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal reference cases: %v", err)
	}

	for i, tc := range cases {
		got := bpe.Encode(nil, tc.Text)
		if !reflect.DeepEqual(got, tc.IDs) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.Text, got, tc.IDs)
		}
	}
}

func TestEncode_Bekko_ByteFallback(t *testing.T) {
	bpe := loadBekkoBPE(t)
	input := "ࡰ"
	ids := bpe.Encode(nil, input)
	decoded := bpe.Decode(ids)
	expectedDecoded := " ࡰ"
	if decoded != expectedDecoded {
		t.Errorf("Decode(Encode(%q)) = %q, want %q", input, decoded, expectedDecoded)
	}
}
