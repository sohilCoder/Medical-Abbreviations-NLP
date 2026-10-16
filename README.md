# Medical Abbreviation Disambiguator

Clinical notes are full of abbreviations that mean different things in different places. "MS" can mean multiple sclerosis, mitral stenosis or morphine sulfate. This program reads the words around an abbreviation and works out which meaning is most likely. It shows the clue words behind its answer, and says "not sure" when the clues are weak.

Written in plain Python with no libraries. It uses a Naive Bayes classifier built from scratch.

```
$ python abbrev.py predict examples/fake_data.txt MS "Give MS 2 mg IV for pain."
MS = morphine sulfate (100% confident)
clues: pain, mg, iv

$ python abbrev.py predict examples/fake_data.txt MS "Patient has MS."
MS: not sure (best guess 'multiple sclerosis', 33%)
clues: (none)
```

## How to run it

Needs Python 3. Run these from the project folder.

**With the made-up sample data** (included):
```
python abbrev.py evaluate examples/fake_data.txt
python abbrev.py predict  examples/fake_data.txt MS "Echo shows MS with a narrow valve."
python test_abbrev.py
```

**With the real dataset:**
1. Download the anonymized data set from the [Clinical Abbreviation Sense Inventory](https://conservancy.umn.edu/items/6651323b-444a-479e-a41a-abca58c2e721) (University of Minnesota).
2. Put the file in a `data/` folder. It isn't included in this repository.
3. Run:
```
python abbrev.py evaluate data/YOUR_FILE.txt
python abbrev.py predict  data/YOUR_FILE.txt AB "Blood type is AB positive."
```

## How it works

1. **Read each example.** Each line of the dataset holds an abbreviation, its correct meaning, its position in the text, the note section, and the note text. The position matters because the same abbreviation can appear twice in one note with different meanings.
2. **Take the context:** the 6 words before and after the abbreviation, plus the section name. Common words like "the" and "had" are skipped.
3. **Train:** for each abbreviation, count how often each meaning occurs and how often each word appears near each meaning.
4. **Predict:** score each meaning as P(meaning) × P(each nearby word | meaning), the Naive Bayes formula. Logs are added instead of multiplying many tiny numbers, and every count gets +1 ("smoothing") so a word never seen before doesn't make a score zero.
5. **Explain:** the clues are the nearby words that are most likely with the chosen meaning compared with the others.
6. **Abstain:** if the best meaning is under 80% likely, it answers "not sure".

## Evaluation

`evaluate` trains on 80% of the examples and tests on the other 20%. It reports:
- **The baseline:** accuracy from always guessing each abbreviation's most common meaning. The model has to beat this to show it learned anything.
- **Accuracy vs. answers given:** how accurate the model is when it's only allowed to answer at higher confidence levels. Saying "not sure" more often should make the remaining answers more accurate.

Results on the made-up sample data are near 100% because it's deliberately simple. The real dataset is the meaningful test.

## Tests

```
python test_abbrev.py
```

12 tests covering text cleaning, the context window, using the position to find the right occurrence, training counts, probabilities, predictions, abstaining, smoothing, and beating the baseline.

## Limitations

- Naive Bayes treats every word as independent, so its confidence is often too high. 100% doesn't really mean certain.
- It only knows the abbreviations and meanings in the training data.
- It's a learning project, not a clinically validated tool.

## Data credit

Moon, S., Pakhomov, S., & Melton, G. (2012). *Clinical Abbreviation Sense Inventory*. University Digital Conservancy. https://hdl.handle.net/11299/137703

`examples/fake_data.txt` is entirely made up.
