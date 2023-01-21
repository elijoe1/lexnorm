from lexnorm.models.annotation import list_eligible


def test_list_eligible():
    assert list_eligible(
        [
            "asdf",
            "asdflj*&",
            "asdfj2342",
            "31kls",
            "asdf lkj",
            "asd''",
            "can''t",
            '"asd',
        ]
    ) == [True, False, True, True, False, True, True, False]
    assert list_eligible(["rt", "rt", "@test", "rt", "test", "@test", "rt"]) == [
        False,
        False,
        False,
        True,
        True,
        False,
        True,
    ]
