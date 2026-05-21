from app.services.text_chunker import TextChunker


def test_single_complete_sentence_in_one_feed():
    chunker = TextChunker()
    assert chunker.feed("Hello world. ") == ["Hello world."]


def test_sentence_split_across_two_feeds():
    chunker = TextChunker()
    assert chunker.feed("Hello wor") == []
    assert chunker.feed("ld. ") == ["Hello world."]


def test_multiple_sentences_in_one_feed():
    chunker = TextChunker()
    assert chunker.feed("Hi. Bye. ") == ["Hi.", "Bye."]


def test_hindi_danda_sentence_end():
    chunker = TextChunker()
    assert chunker.feed("नमस्ते। ") == ["नमस्ते।"]


def test_trailing_sentence_ender_waits_for_more_text():
    chunker = TextChunker()
    # "Hello." with nothing after it yet -- could still become "Hello.World"
    # -- so we must NOT split until whitespace confirms the sentence ended.
    assert chunker.feed("Hello.") == []
    assert chunker.feed(" World.") == ["Hello."]
    assert chunker.flush() == "World."


def test_decimal_number_is_not_split():
    chunker = TextChunker()
    assert chunker.feed("Pi is 3.14 approx. ") == ["Pi is 3.14 approx."]


def test_ellipsis_and_multi_punctuation_treated_as_one_boundary():
    chunker = TextChunker()
    assert chunker.feed("Wait... really?! Yes. ") == ["Wait...", "really?!", "Yes."]


def test_max_length_forced_flush_with_no_punctuation():
    chunker = TextChunker(max_chars=10)
    chunks = chunker.feed("this is a long run-on clause with no punctuation at all")
    assert chunks == ["this is a long run-on clause with no punctuation at all"]


def test_max_length_flush_only_triggers_past_threshold():
    chunker = TextChunker(max_chars=100)
    assert chunker.feed("short text, no sentence end") == []


def test_flush_returns_leftover_text():
    chunker = TextChunker()
    assert chunker.feed("Hello there") == []
    assert chunker.flush() == "Hello there"


def test_flush_returns_none_when_buffer_empty():
    chunker = TextChunker()
    assert chunker.flush() is None


def test_flush_returns_none_after_all_sentences_already_emitted():
    chunker = TextChunker()
    assert chunker.feed("Complete sentence. ") == ["Complete sentence."]
    assert chunker.flush() is None
