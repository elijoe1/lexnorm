from lexnorm.models.filtering import is_eligible


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
    # assert list_eligible(["rt", "rt", "@test", "rt", "test", "@test", "rt"]) == [
    #     False,
    #     False,
    #     False,
    #     True,
    #     True,
    #     False,
    #     True,
    # ]
    assert [
        is_eligible(c)
        for c in ["can't", "wo'nt", "''", "'", "asdklj'''as", "asd'", "'sadf'", "'dsa"]
    ] == [True, True, False, False, True, False, False, False]
