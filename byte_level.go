package tokenizer

// GPT-2 / HuggingFace ByteLevel mapping tables.
// Maps raw byte (0..255) to a unique printable Unicode rune.

var byteToRuneTable = [256]rune{
	256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
	272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
	288, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
	64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
	80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
	96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
	112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 289,
	290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
	306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
	322, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 323, 174, 175,
	176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
	192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
	208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
	224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
	240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
}

var runeToByteTable [324]byte

func init() {
	for b, r := range byteToRuneTable {
		if r >= 0 && int(r) < len(runeToByteTable) {
			runeToByteTable[r] = byte(b)
		}
	}
}

// byteToRune returns the byte-level representation rune for a byte.
func byteToRune(b byte) rune {
	return byteToRuneTable[b]
}

// runeToByte returns the original byte for a byte-level rune.
func runeToByte(r rune) (byte, bool) {
	if r >= 0 && int(r) < len(runeToByteTable) {
		return runeToByteTable[r], true
	}
	return 0, false
}

// encodeBytes converts a slice of bytes into its byte-level string representation.
func encodeBytes(bs []byte) string {
	runes := make([]rune, len(bs))
	for i, b := range bs {
		runes[i] = byteToRuneTable[b]
	}
	return string(runes)
}

// ByteLevelScheme is granite-embedding-97m-multilingual-r2's pretokenizer: the reference
// regex (pretokenize.go) plus GPT-2 byte-level encode/decode.
type ByteLevelScheme struct{}

func (ByteLevelScheme) Pretokenize(text string) []string {
	return pretokenize(text)
}

func (ByteLevelScheme) ByteFallbackSymbol(b byte) (string, bool) {
	return "", false
}

func (ByteLevelScheme) DecodeToken(dst []byte, tok string) []byte {
	for _, r := range tok {
		if b, ok := runeToByte(r); ok {
			dst = append(dst, b)
		} else {
			dst = append(dst, []byte(string(r))...)
		}
	}
	return dst
}
