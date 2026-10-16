# test_abbrev.py - tests for abbrev.py
#
# Run with:  python test_abbrev.py
# Uses the made-up examples/fake_data.txt, so it works without the real dataset.

from abbrev import CONFIDENT, clean_words, clues, context, load, predict, split, train

FAKE = "examples/fake_data.txt"


def guess(model, abbr, sentence):
    """Returns (meaning, probability) for the first time abbr appears in sentence."""
    before, after = sentence.split(abbr, 1)
    return predict(model, abbr, context(before, after))[0]


# ----- reading the data -----

def test_clean_words():
    assert clean_words("Blood type: AB-positive!") == ["blood", "type", "abpositive"]
    assert clean_words("_%#NAME#%_ _%#NAME#%_ is 29") == ["29"]          # privacy tags and "is" removed
    assert clean_words("The patient had a missed AB") == ["missed", "ab"]  # common words removed


def test_context_takes_words_on_each_side():
    before = "one two three four five six seven eight"
    after = "nine ten eleven twelve thirteen fourteen fifteen"
    assert context(before, after) == ["three", "four", "five", "six", "seven", "eight",
                                      "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]


def test_position_picks_the_right_occurrence():
    # Two "AB"s with different meanings. The positions say which one is labeled.
    text = "History of 1 spontaneous AB. She is AB positive."
    start = text.index("AB")                     # the first one
    words = context(text[:start], text[start + 2:])
    assert "spontaneous" in words


def test_load():
    examples = load(FAKE)
    assert len(examples) == 175
    first = examples[0]
    assert first["abbr"] == "MS" and first["meaning"] == "multiple sclerosis"
    assert "relapsing" in first["words"]
    assert "section_history" in first["words"]   # the note section is a clue too


# ----- training and predicting -----

def test_train_counts_meanings():
    model = train(load(FAKE))
    assert sorted(model) == ["MS", "PT", "RA"]
    assert sorted(model["MS"]["meaning_counts"]) == ["mitral stenosis", "morphine sulfate", "multiple sclerosis"]
    assert sum(model["MS"]["meaning_counts"].values()) == 75


def test_probabilities_add_up_to_one_and_are_sorted():
    model = train(load(FAKE))
    results = predict(model, "MS", ["valve", "echo"])
    total = 0
    for meaning, probability in results:
        total += probability
    assert abs(total - 1) < 0.000001
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_clear_sentences_get_the_right_meaning():
    model = train(load(FAKE))
    assert guess(model, "MS", "Echo shows severe MS with a narrow valve.")[0] == "mitral stenosis"
    assert guess(model, "MS", "Give MS 4 mg IV for pain.")[0] == "morphine sulfate"
    assert guess(model, "MS", "MRI shows lesions from her MS, neurology following.")[0] == "multiple sclerosis"
    assert guess(model, "PT", "Labs: PT 19 seconds, INR 2.8.")[0] == "prothrombin time"
    assert guess(model, "PT", "Start PT for gait training.")[0] == "physical therapy"


def test_not_sure_when_there_are_no_clues():
    model = train(load(FAKE))
    meaning, probability = guess(model, "MS", "Patient has MS.")
    assert probability < CONFIDENT


def test_words_never_seen_before_dont_break_it():
    model = train(load(FAKE))
    results = predict(model, "RA", ["zebra", "spaceship"])   # smoothing keeps probabilities above 0
    assert len(results) == 2


def test_clues_point_to_the_chosen_meaning():
    model = train(load(FAKE))
    words = context("Pain controlled with", "4 mg IV every 4 hours")
    assert "mg" in clues(model, "MS", words, "morphine sulfate")


# ----- evaluation -----

def test_split_is_80_20_and_repeatable():
    examples = load(FAKE)
    training, testing = split(examples)
    assert len(training) == 140 and len(testing) == 35
    assert split(examples)[1] == testing          # same split every time


def test_beats_always_guessing_the_most_common_meaning():
    training, testing = split(load(FAKE))
    model = train(training)
    right = 0
    baseline_right = 0
    for ex in testing:
        if predict(model, ex["abbr"], ex["words"])[0][0] == ex["meaning"]:
            right += 1
        counts = model[ex["abbr"]]["meaning_counts"]
        if max(counts, key=counts.get) == ex["meaning"]:
            baseline_right += 1
    assert right > baseline_right


# ----- run every test -----

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name in sorted(dir()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print("PASS", name)
                passed += 1
            except Exception as error:
                print("FAIL", name, "-", repr(error))
                failed += 1
    print()
    print(passed, "passed,", failed, "failed")
