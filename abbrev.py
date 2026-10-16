# abbrev.py - figure out what a medical abbreviation means from the words around it
#
# Example: in "blood type is AB positive", AB means "blood group",
#          in "she had a missed AB at 6 weeks", AB means "abortion".
#
# How it works (Naive Bayes):
#   1. For every training example, take the words just before and after the abbreviation.
#   2. Count how often each word appears near each meaning.
#   3. For a new sentence, score each meaning by how well its word counts fit the
#      nearby words. The highest score wins. If it isn't confident, it says "not sure".
#
# How to run it:
#   python abbrev.py evaluate DATA_FILE
#   python abbrev.py predict  DATA_FILE ABBREVIATION "a sentence that uses it"

import math
import random
import sys

WINDOW = 6          # how many words to look at on each side of the abbreviation
CONFIDENT = 0.80    # below this probability, answer "not sure"

# Very common words that appear near every meaning, so they aren't useful clues.
COMMON_WORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by",
                "is", "was", "are", "were", "be", "been", "has", "had", "have", "she", "he",
                "her", "his", "who", "this", "that", "as", "from", "patient"}


# ---------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------

def clean_words(text):
    """Lowercase words with punctuation removed.
    Skips the _%#NAME#%_ privacy tags and very common words."""
    words = []
    for word in text.split():
        if "%#" in word:
            continue
        letters = ""
        for ch in word.lower():
            if ch.isalnum():
                letters += ch
        if letters != "" and letters not in COMMON_WORDS:
            words.append(letters)
    return words


def context(text_before, text_after):
    """The WINDOW words just before and just after the abbreviation."""
    before = clean_words(text_before)[-WINDOW:]
    after = clean_words(text_after)[:WINDOW]
    return before + after


def load(filename):
    """Reads the CASI data file. Each line looks like:
       abbreviation|meaning|as written|start|end|section|note text
    Returns a list of examples: {"abbr": ..., "meaning": ..., "words": [...]}"""
    examples = []
    with open(filename, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 6)
            if len(parts) != 7:
                continue                          # skip broken lines
            abbr, meaning, written, start, end, section, text = parts
            if not start.isdigit() or not end.isdigit():
                continue
            start = int(start)
            end = int(end)
            words = context(text[:start], text[end + 1:])
            for word in clean_words(section):     # the note section is a clue too
                words.append("section_" + word)
            examples.append({"abbr": abbr, "meaning": meaning, "words": words})
    return examples


# ---------------------------------------------------------------------------
# Training: count words near each meaning
# ---------------------------------------------------------------------------

def train(examples):
    """Builds a model: for each abbreviation, how often each meaning occurs,
    and how often each word appears near each meaning."""
    model = {}
    for ex in examples:
        if ex["abbr"] not in model:
            model[ex["abbr"]] = {"meaning_counts": {}, "word_counts": {}, "word_totals": {},
                                 "all_words": set()}
        m = model[ex["abbr"]]

        meaning = ex["meaning"]
        m["meaning_counts"][meaning] = m["meaning_counts"].get(meaning, 0) + 1
        if meaning not in m["word_counts"]:
            m["word_counts"][meaning] = {}
            m["word_totals"][meaning] = 0
        for word in ex["words"]:
            counts = m["word_counts"][meaning]
            counts[word] = counts.get(word, 0) + 1
            m["word_totals"][meaning] += 1       # total words seen near this meaning
            m["all_words"].add(word)
    return model


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------

def word_probability(m, meaning, word):
    """How likely this word is to appear near this meaning.
    Adds 1 to every count ("smoothing") so an unseen word doesn't make the probability 0."""
    count = m["word_counts"][meaning].get(word, 0)
    return (count + 1) / (m["word_totals"][meaning] + len(m["all_words"]) + 1)


def predict(model, abbr, words):
    """Returns a list of (meaning, probability), most likely first."""
    m = model[abbr]
    total_examples = sum(m["meaning_counts"].values())

    # Score = log P(meaning) + sum of log P(word | meaning).
    # Logs are added instead of multiplying many tiny numbers, which would round to 0.
    scores = {}
    for meaning in m["meaning_counts"]:
        score = math.log(m["meaning_counts"][meaning] / total_examples)
        for word in words:
            score += math.log(word_probability(m, meaning, word))
        scores[meaning] = score

    # Turn the scores into probabilities that add up to 1.
    best_score = max(scores.values())
    total = 0
    for meaning in scores:
        total += math.exp(scores[meaning] - best_score)
    results = []
    for meaning in scores:
        results.append((meaning, math.exp(scores[meaning] - best_score) / total))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


def clues(model, abbr, words, meaning, how_many=3):
    """The nearby words that pointed most strongly toward this meaning."""
    m = model[abbr]
    strength = {}
    for word in set(words):
        here = word_probability(m, meaning, word)
        elsewhere = 0
        for other in m["meaning_counts"]:
            if other != meaning:
                elsewhere = max(elsewhere, word_probability(m, other, word))
        if elsewhere > 0 and here > elsewhere:
            strength[word] = here / elsewhere      # how many times likelier with this meaning
    ranked = sorted(strength, key=strength.get, reverse=True)
    return ranked[:how_many]


def explain(model, abbr, sentence):
    """Finds abbr in the sentence and prints what it most likely means, and why."""
    if abbr not in model:
        print("I don't know the abbreviation '" + abbr + "'.")
        return

    raw = sentence.split()
    position = -1
    for i in range(len(raw)):
        if raw[i].strip(".,;:()") == abbr:
            position = i
            break
    if position == -1:
        print("'" + abbr + "' doesn't appear in that sentence.")
        return

    words = context(" ".join(raw[:position]), " ".join(raw[position + 1:]))
    results = predict(model, abbr, words)
    best, probability = results[0]

    if probability < CONFIDENT:
        print(abbr + ": not sure (best guess '" + best + "', " + str(round(probability * 100)) + "%)")
    else:
        print(abbr + " = " + best + " (" + str(round(probability * 100)) + "% confident)")
    found = clues(model, abbr, words, best)
    if len(found) == 0:
        print("clues: (none)")
    else:
        print("clues: " + ", ".join(found))
    print("all options:")
    for meaning, p in results:
        print("  " + str(round(p * 100)).rjust(3) + "%  " + meaning)


# ---------------------------------------------------------------------------
# Evaluating: train on 80% of the data, test on the other 20%
# ---------------------------------------------------------------------------

def split(examples, test_fraction=0.2):
    shuffled = examples[:]
    random.seed(0)                              # the same split every run
    random.shuffle(shuffled)
    cut = int(len(shuffled) * test_fraction)
    return shuffled[cut:], shuffled[:cut]       # (training, testing)


def evaluate(examples):
    training, testing = split(examples)
    model = train(training)

    # Baseline to beat: always guess each abbreviation's most common meaning.
    baseline_right = 0
    results = []                                # (was it right?, confidence) per test example
    for ex in testing:
        if ex["abbr"] not in model:
            continue
        counts = model[ex["abbr"]]["meaning_counts"]
        most_common = max(counts, key=counts.get)
        if most_common == ex["meaning"]:
            baseline_right += 1
        best, probability = predict(model, ex["abbr"], ex["words"])[0]
        results.append((best == ex["meaning"], probability))

    tested = len(results)
    print("trained on", len(training), "examples, tested on", tested)
    print("baseline (always guess the most common meaning):", percent(baseline_right, tested))
    print()
    print("when allowed to say 'not sure':")
    print("  confidence needed   answers given   accuracy of those answers")
    for threshold in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        answered = 0
        right = 0
        for was_right, probability in results:
            if probability >= threshold:
                answered += 1
                if was_right:
                    right += 1
        print("  " + (str(round(threshold * 100)) + "%").ljust(20)
              + percent(answered, tested).ljust(16) + percent(right, answered))


def percent(part, whole):
    if whole == 0:
        return "-"
    return str(round(100 * part / whole, 1)) + "%"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "evaluate":
        evaluate(load(args[1]))
    elif len(args) == 4 and args[0] == "predict":
        explain(train(load(args[1])), args[2], args[3])
    else:
        print("usage:")
        print('  python abbrev.py evaluate DATA_FILE')
        print('  python abbrev.py predict  DATA_FILE ABBREVIATION "a sentence that uses it"')


if __name__ == "__main__":
    main()
