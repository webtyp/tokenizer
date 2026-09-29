"""Reference pre-tokenization of Qwen3.5 (NFC normalizer + regex Split + ByteLevel), from the
model's own tokenizer.json via Hugging Face `tokenizers`. Run with the reference environment:

    ~/Dev/LMmodels/.venv/bin/python testdata/gen_qwen35_pretokenize.py ~/Dev/LMmodels/Qwen/Qwen3.5-0.8B
"""
import json, os, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.path.join(sys.argv[1], "tokenizer.json"))
cases = [
    "Hola mundo", "¿A qué hora abren los lunes?", "Señor Muñoz — ítem “café orgánico” ¿1.250 €? ¡Sí!",
    "é decomposed é", "ñ and Ñ", "I'm here, they've gone, it's OK", "DON'T SHOUT",
    "x=12345; y = 3.14159", "  leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\n\n\nend",
    "emoji 😀 ok", "日本語のテキスト", "code: func main() { fmt.Println(\"hi\") }", "a\r\nb\rc",
    "<tool_call>\n<function=clinic_hours>\n<parameter=day>\nlunes\n</parameter>\n</function>\n</tool_call>",
    "", " ", "123abc456", "¡¡¡!!!", "C'est déjà l'été",
]
out = []
for text in cases:
    normalized = tok.normalizer.normalize_str(text) if tok.normalizer else text
    pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(normalized)]
    out.append({"text": text, "normalized": normalized, "pretokens": pieces,
                "ids": tok.encode(text, add_special_tokens=False).ids})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35_pretokenize.json"), "w"), ensure_ascii=False, indent=1)
print("cases", len(out))
