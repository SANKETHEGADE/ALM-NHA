import json
import os
import unicodedata
from typing import List, Dict, Optional, Union
from src.asr.preprocessing import normalize_text, normalize_language_code, SUPPORTED_LANGUAGES

# Standard Special Tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BLANK_TOKEN = "<blank>"  # CTC Blank
SPACE_TOKEN = "|"

# Language Identifiers
LANG_TOKENS = [f"<|{lang}|>" for lang in SUPPORTED_LANGUAGES]

# Basic character sets for 7 target languages
# English (Latin letters, digits, basic apostrophe)
EN_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789'")

# Hindi (Devanagari Unicode block U+0900 to U+097F)
HI_CHARS = [chr(i) for i in range(0x0900, 0x097F + 1)]

# Telugu (Telugu Unicode block U+0C00 to U+0C7F)
TE_CHARS = [chr(i) for i in range(0x0C00, 0x0C7F + 1)]

# Tamil (Tamil Unicode block U+0B80 to U+0BFF)
TA_CHARS = [chr(i) for i in range(0x0B80, 0x0BFF + 1)]

# Bengali (Bengali Unicode block U+0980 to U+09FF)
BN_CHARS = [chr(i) for i in range(0x0980, 0x09FF + 1)]

# Urdu (Arabic & Arabic Supplement & Extended Unicode blocks used in Urdu)
UR_CHARS = [chr(i) for i in range(0x0600, 0x06FF + 1)] + [chr(i) for i in range(0x0750, 0x077F + 1)]

# Mandarin (Top frequency common Hanzi & Pinyin components)
# A wide base of common CJK Unified Ideographs
ZH_BASE_CHARS = [
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形与关各",
    "警救助火车危急害伤抢刀枪病医死险爆电跑逃避快听静谁何么哪怎救命报警警察医生火灾危险救我帮忙打架车祸地震爆炸",
]


class MultilingualTokenizer:
    """
    Unified Non-Whisper Multilingual Tokenizer for 7 Target Languages:
    Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, and English.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.blank_token = BLANK_TOKEN
        self.space_token = SPACE_TOKEN
        self.lang_tokens = LANG_TOKENS

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Construct deterministic multilingual vocabulary with special and script tokens."""
        vocab = [
            self.pad_token,    # 0
            self.unk_token,    # 1
            self.blank_token,  # 2 (CTC blank)
            self.space_token,  # 3
        ] + self.lang_tokens   # 4..10

        # Add language script characters
        seen = set(vocab)
        all_chars = EN_CHARS + HI_CHARS + TE_CHARS + TA_CHARS + BN_CHARS + UR_CHARS
        for block in ZH_BASE_CHARS:
            all_chars.extend(list(block))

        for ch in all_chars:
            if ch not in seen and not unicodedata.category(ch).startswith("C"):
                vocab.append(ch)
                seen.add(ch)

        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def blank_id(self) -> int:
        return self.token_to_id[self.blank_token]

    @property
    def space_id(self) -> int:
        return self.token_to_id[self.space_token]

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size()

    def get_lang_token_id(self, language: str) -> int:
        lang = normalize_language_code(language)
        token = f"<|{lang}|>"
        return self.token_to_id.get(token, self.unk_id)

    def encode(
        self,
        text: str,
        language: Optional[str] = None,
        add_special_tokens: bool = False
    ) -> List[int]:
        """
        Encode text into a list of integer token IDs.
        """
        if not text:
            return []

        lang = normalize_language_code(language) if language else "en"
        norm_text = normalize_text(text, language=lang)

        token_ids: List[int] = []

        if add_special_tokens and lang:
            lang_id = self.get_lang_token_id(lang)
            if lang_id != self.unk_id:
                token_ids.append(lang_id)

        for char in norm_text:
            if char == " ":
                token_ids.append(self.space_id)
            elif char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                token_ids.append(self.unk_id)

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode a sequence of integer token IDs into text.
        """
        chars = []
        special_ids = {self.pad_id, self.unk_id, self.blank_id}
        if skip_special_tokens:
            for lt in self.lang_tokens:
                if lt in self.token_to_id:
                    special_ids.add(self.token_to_id[lt])

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid == self.space_id:
                chars.append(" ")
            elif tid in self.id_to_token:
                token = self.id_to_token[tid]
                if skip_special_tokens and token.startswith("<|") and token.endswith("|>"):
                    continue
                chars.append(token)
            else:
                if not skip_special_tokens:
                    chars.append(self.unk_token)

        text = "".join(chars)
        # Clean up repeated whitespace
        return " ".join(text.split()).strip()

    def batch_encode(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None,
        add_special_tokens: bool = False
    ) -> List[List[int]]:
        """Encode a batch of texts."""
        if languages is None:
            languages = ["en"] * len(texts)
        elif len(languages) != len(texts):
            languages = [languages[0]] * len(texts)

        return [
            self.encode(text, language=lang, add_special_tokens=add_special_tokens)
            for text, lang in zip(texts, languages)
        ]

    def batch_decode(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_token_ids]

    def save_vocab(self, save_path: str):
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, vocab_path: str):
        """Load vocabulary from JSON file."""
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
