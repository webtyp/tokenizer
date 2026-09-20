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
		Vocab:  cfg.Vocab,
		Merges: cfg.Merges,
	})
	if err != nil {
		t.Fatalf("failed to create BPE: %v", err)
	}
	return bpe
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
	if ids[0] != BosTokenID {
		t.Errorf("expected first ID to be BOS (%d), got %d", BosTokenID, ids[0])
	}
	if ids[len(ids)-1] != EosTokenID {
		t.Errorf("expected last ID to be EOS (%d), got %d", EosTokenID, ids[len(ids)-1])
	}
}

func TestEncode_EmptyString(t *testing.T) {
	bpe := loadTestBPE(t)
	ids := bpe.Encode(nil, "")
	want := []int32{BosTokenID, EosTokenID}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("got %v, want %v", ids, want)
	}
}

func TestIgnoreMerges_WholeWordInVocab(t *testing.T) {
	bpe := loadTestBPE(t)

	// "Hola" is in vocab (ID 47970).
	// With ignore_merges: true, pretoken "Hola" is encoded as a single token without merge iteration.
	ids := bpe.Encode(nil, "Hola")
	want := []int32{BosTokenID, 47970, EosTokenID}
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
