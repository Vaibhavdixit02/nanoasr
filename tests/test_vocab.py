from nanoasr.vocab import (
    VOCAB, VOCAB_SIZE, BLANK_IDX, char_to_idx, idx_to_char,
    encode, decode_indices,
)


def test_vocab_size():
    assert len(VOCAB) == 28
    assert VOCAB_SIZE == 28


def test_blank_is_last():
    assert BLANK_IDX == 27
    assert VOCAB[27] == "<blank>"


def test_space_index():
    assert char_to_idx[" "] == 26
    assert idx_to_char[26] == " "


def test_encode_hello_world():
    indices = encode("hello world")
    assert indices == [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]


def test_encode_skips_unknown_chars():
    assert encode("hi!") == [7, 8]
    assert encode("test123") == [19, 4, 18, 19]
    assert encode("") == []


def test_decode_roundtrip():
    text = "the quick brown fox"
    assert decode_indices(encode(text)) == text


def test_decode_skips_blank():
    assert decode_indices([7, BLANK_IDX, 8]) == "hi"


def test_all_letters_mapped():
    for c in "abcdefghijklmnopqrstuvwxyz ":
        assert c in char_to_idx
        idx = char_to_idx[c]
        assert idx_to_char[idx] == c
