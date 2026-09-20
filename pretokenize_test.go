package tokenizer

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

type pretokenizeCase struct {
	Input string   `json:"input"`
	Want  []string `json:"want"`
}

func TestPretokenizeAgainstReference(t *testing.T) {
	data, err := os.ReadFile("testdata/pretokenize_reference.json")
	if err != nil {
		t.Fatalf("failed to read testdata/pretokenize_reference.json: %v", err)
	}

	var cases []pretokenizeCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	for i, tc := range cases {
		got := pretokenize(tc.Input)
		if !reflect.DeepEqual(got, tc.Want) {
			t.Errorf("case %d (%q):\n got:  %v\n want: %v", i, tc.Input, got, tc.Want)
		}
	}
}

func TestPretokenize_TrailingWhitespace(t *testing.T) {
	input := "hello   "
	want := []string{"hello", "ĠĠĠ"}
	got := pretokenize(input)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}
