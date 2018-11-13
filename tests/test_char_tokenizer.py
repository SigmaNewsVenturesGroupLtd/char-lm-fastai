from model import CharTokenizer


def test_should_tokenize_chars():
    # GIVEN
    text = "xxfld 1 hello"
    # WHEN
    res = CharTokenizer().process_all([text])[0]
    # THEN
    assert res == ['bos', 'h', 'e', 'l', 'l', 'o']