from lexnorm.generate_extract.filtering import is_eligible


def test_is_eligible():
    assert [
        is_eligible(c)
        for c in [
            "asdf",
            "asdflj*&",
            "asdfj2342",
            "31kls",
            "asdf lkj",
            "asd''",
            "can''t",
            '"asd',
        ]
    ] == [True, False, True, True, True, False, True, False]
    assert not any([is_eligible(c) for c in ["茶", "<S>", "/s", "</s>", "<s> the"]])
    assert all(
        [
            is_eligible(c, allow_special=True)
            for c in ["<S>", "<s>", "</S>", "</s>", "<S> the", "the </s>"]
        ]
    )
    assert not any(
        is_eligible(c, allow_special=True) for c in ["<<s> the", "<s> </s>", "t<S>he"]
    )
    assert is_eligible("abcdefghijklmnopqrstuvwxyz' 012456789")
    assert [
        is_eligible(c)
        for c in ["can't", "wo'nt", "''", "'", "asdklj'''as", "asd'", "'sadf'", "'dsa"]
    ] == [True, True, False, False, True, False, False, False]
