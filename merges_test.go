package tokenizer

import (
	"reflect"
	"testing"
)

func TestParseMerges(t *testing.T) {
	cases := []struct {
		name string
		data string
		want []string
	}{
		{"unix line ends", "Ġ t\nĠ a\n", []string{"Ġ t", "Ġ a"}},
		{"windows line ends", "Ġ t\r\nĠ a\r\n", []string{"Ġ t", "Ġ a"}},
		{"no final line end", "Ġ t\nĠ a", []string{"Ġ t", "Ġ a"}},
		{"empty lines skipped", "Ġ t\n\nĠ a\n", []string{"Ġ t", "Ġ a"}},
		{"empty file", "", nil},
	}
	for _, c := range cases {
		if got := ParseMerges([]byte(c.data)); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: ParseMerges = %q, want %q", c.name, got, c.want)
		}
	}
}
