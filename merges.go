package tokenizer

// ParseMerges reads a .merges file (webtyp/weightsc's companion output: one "left right" pair per
// line, in rank order) into Config.Merges. Line ends may be "\n" or "\r\n"; empty lines are skipped.
func ParseMerges(data []byte) []string {
	var merges []string
	start := 0
	for i := 0; i <= len(data); i++ {
		if i < len(data) && data[i] != '\n' {
			continue
		}
		line := data[start:i]
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}
		if len(line) > 0 {
			merges = append(merges, string(line))
		}
		start = i + 1
	}
	return merges
}
