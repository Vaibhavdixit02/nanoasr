from nanoasr.vocab import clean_text


def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"


def test_clean_text_strips_punctuation():
    assert clean_text("Hello, World!") == "hello world"


def test_clean_text_collapses_spaces():
    assert clean_text("a  b   c") == "a b c"


def test_clean_text_strips_numbers():
    assert clean_text("test 123 abc") == "test abc"
    assert clean_text("test123abc") == "testabc"


def test_clean_text_empty():
    assert clean_text("") == ""


def test_clean_text_only_punctuation():
    assert clean_text("!!!???") == ""
