package tokenizer

import (
	"reflect"
	"testing"
)

// A character split across tokens is shown once it completes; ASCII is shown at once.
func TestStream_HoldsBackHalfCharacters(t *testing.T) {
	var s Stream
	var got []string
	for _, tok := range [][]byte{[]byte("ma"), {0xC3}, {0xB1}, []byte("ana "), {0xE3, 0x81}, {0x97}} {
		got = append(got, s.Write(tok))
	}
	want := []string{"ma", "", "ñ", "ana ", "", "し"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Write results %q, want %q", got, want)
	}
	if s.Text() != "mañana し" || s.Flush() != "" {
		t.Fatalf("Text %q", s.Text())
	}
}

// A tail that never completes is held while text keeps coming, and Flush shows it at the end:
// the pieces joined are always Text.
func TestStream_FlushShowsAnIncompleteTail(t *testing.T) {
	var s Stream
	var got []string
	for _, tok := range [][]byte{[]byte("しました"), {0xE9, 0x8C}, {0xE9, 0x8C}, []byte(" ll"), {0xE9}} {
		got = append(got, s.Write(tok))
	}
	got = append(got, s.Flush())
	want := []string{"しました", "", "\xe9\x8c", "\xe9\x8c ll", "", "\xe9"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("pieces %q, want %q", got, want)
	}
	joined := ""
	for _, p := range got {
		joined += p
	}
	if joined != s.Text() {
		t.Fatalf("joined %q, Text %q", joined, s.Text())
	}
}
