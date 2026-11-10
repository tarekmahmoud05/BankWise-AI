from bankwise_rag.normalize import detect_language, normalize_text


def test_arabic_digits_become_ascii():
    assert normalize_text("قرض ٥٠٬٠٠٠ جنيه لمدة ٣٦ شهر") == "قرض 50,000 جنيه لمدة 36 شهر"


def test_diacritics_tatweel_and_alef_variants():
    assert normalize_text("الإِجَابَـــة") == "الاجابة"
    assert normalize_text("أحمد إبراهيم آمن") == "احمد ابراهيم امن"


def test_alef_maqsura_whitespace_and_invisible_chars():
    assert normalize_text("على‏   مستوى\n\n الفرع") == "علي مستوي الفرع"


def test_ta_marbuta_is_preserved():
    assert "بطاقة" in normalize_text("بطاقة")


def test_empty_and_none_safe():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_detect_language():
    assert detect_language("ما هي رسوم القرض") == "ar"
    assert detect_language("What is the fee") == "en"
