from app.services.speech_sanitizer import sanitize_for_speech


def test_real_world_neural_network_example_end_to_end():
    text = (
        "A **neural network** is a sorted triple $(N, V, w)$ with two sets $N, V$ "
        "and a function $w$, where $N$ is the set of *neurons* and $V$ a set "
        "$\\{(i, j) | i, j \\in N\\}$ whose elements are called *connections* "
        "between neuron $i$ and neuron $j$."
    )
    assert sanitize_for_speech(text) == (
        "A neural network is a sorted triple (N, V, w) with two sets N, V and a "
        "function w, where N is the set of neurons and V a set (i, j) such that "
        "i, j in N whose elements are called connections between neuron i and "
        "neuron j."
    )


def test_bold_double_star_is_stripped():
    assert sanitize_for_speech("This is **important** text.") == "This is important text."


def test_bold_double_underscore_is_stripped():
    assert sanitize_for_speech("This is __important__ text.") == "This is important text."


def test_italic_single_star_is_stripped():
    assert sanitize_for_speech("This is *important* text.") == "This is important text."


def test_italic_single_underscore_is_stripped():
    assert sanitize_for_speech("This is _important_ text.") == "This is important text."


def test_snake_case_identifier_not_treated_as_italic():
    assert sanitize_for_speech("Tune the learning_rate parameter.") == "Tune the learning_rate parameter."


def test_asterisk_used_as_multiplication_is_not_treated_as_italic():
    # No non-whitespace char immediately inside either '*', so this can't be
    # mistaken for an italic span -- without that guard this would wrongly
    # collapse to "3 4 5".
    assert sanitize_for_speech("3 * 4 * 5") == "3 * 4 * 5"


def test_inline_code_backticks_stripped():
    assert sanitize_for_speech("Run `pytest` to test.") == "Run pytest to test."


def test_strikethrough_stripped():
    assert sanitize_for_speech("This is ~~wrong~~ right.") == "This is wrong right."


def test_header_marker_stripped():
    assert sanitize_for_speech("## Section title") == "Section title"


def test_list_marker_stripped():
    assert sanitize_for_speech("- first item") == "first item"


def test_link_markdown_reduced_to_link_text():
    assert sanitize_for_speech("See [the docs](https://example.com) for more.") == "See the docs for more."


def test_inline_dollar_delimiters_removed():
    assert sanitize_for_speech("The value $x$ is positive.") == "The value x is positive."


def test_block_double_dollar_delimiters_removed():
    assert sanitize_for_speech("$$x + y$$ is the sum.") == "x + y is the sum."


def test_frac_converted_to_over():
    assert sanitize_for_speech("The ratio is $\\frac{a}{b}$.") == "The ratio is a over b."


def test_sqrt_converted():
    assert sanitize_for_speech("Compute $\\sqrt{x}$ first.") == "Compute the square root of x first."


def test_superscript_braced_and_bare():
    assert sanitize_for_speech("$x^{n}$ and $x^2$ both appear.") == "x to the n and x to the 2 both appear."


def test_subscript_braced_and_bare():
    assert sanitize_for_speech("$w_{ij}$ and $x_i$ both appear.") == "w sub ij and x sub i both appear."


def test_greek_letters_converted():
    assert sanitize_for_speech("Let $\\alpha$ and $\\beta$ be constants.") == "Let alpha and beta be constants."


def test_set_membership_and_quantifiers():
    assert sanitize_for_speech("$\\forall x \\exists y, x \\in Y$.") == "for all x there exists y, x in Y."


def test_comparison_operators():
    assert sanitize_for_speech("$a \\leq b$ and $c \\geq d$ and $e \\neq f$.") == (
        "a less than or equal to b and c greater than or equal to d and e not equal to f."
    )


def test_times_and_cdot():
    assert sanitize_for_speech("$a \\times b$ equals $c \\cdot d$.") == "a times b equals c times d."


def test_arrow_to_and_rightarrow():
    assert sanitize_for_speech("$f: X \\to Y$ and $g \\rightarrow h$.") == "f: X maps to Y and g maps to h."


def test_ldots():
    assert sanitize_for_speech("$1, 2, \\ldots, n$ are the indices.") == "1, 2, and so on, n are the indices."


def test_escaped_braces_and_pipe_set_builder_notation():
    assert sanitize_for_speech("Let $\\{(i, j) | i, j \\in N\\}$ be the edge set.") == (
        "Let (i, j) such that i, j in N be the edge set."
    )


def test_unrecognized_latex_command_falls_back_safely():
    # \mathbb isn't in the explicit rule table -- must degrade to a readable
    # run-on word instead of leaving stray backslashes/braces for TTS.
    assert sanitize_for_speech("The set $\\mathbb{R}$ is uncountable.") == "The set mathbbR is uncountable."


def test_single_unmatched_dollar_amount_left_untouched():
    # No second '$' in this sentence, so the inline-math regex can't match
    # at all -- OpenAI's TTS already reads a bare "$5" correctly on its own.
    assert sanitize_for_speech("It costs $5.") == "It costs $5."


def test_standalone_bare_number_inline_math_is_converted():
    assert sanitize_for_speech("The value $5$ satisfies the equation.") == "The value 5 satisfies the equation."


def test_compound_currency_in_one_sentence_is_a_known_limitation():
    # Accepted, documented limitation: two independent '$' mentions in one
    # chunk get bridged into a single (incorrectly) converted math span.
    # Not worth guarding against for this math/CS-flavored corpus -- see the
    # module docstring and the plan's Decision Log note.
    assert sanitize_for_speech("It costs $5 today but $10 tomorrow.") == "It costs 5 today but 10 tomorrow."


def test_plain_text_with_ordinary_punctuation_is_unchanged():
    text = "Wait... don't go, it's 3.14% off, right?!"
    assert sanitize_for_speech(text) == text


def test_passthrough_matches_existing_orchestration_fixtures():
    # Regression guard: none of the sentences already asserted verbatim in
    # tests/test_orchestration.py should be altered by the sanitizer.
    fixtures = [
        "Hello world.",
        "Second sentence.",
        "no punctuation here",
        "Hello.",
        "Second.",
        "Grounded answer.",
        "I don't have enough information in the knowledge base to answer that.",
        "Hello!",
        "OLAP is...",
    ]
    for sentence in fixtures:
        assert sanitize_for_speech(sentence) == sentence


def test_idempotent_on_already_sanitized_text():
    text = "A neural network is a sorted triple with two sets N, V and a function w."
    assert sanitize_for_speech(sanitize_for_speech(text)) == sanitize_for_speech(text)
