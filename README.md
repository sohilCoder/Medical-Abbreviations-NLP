clinical notes are full of abbreviations that can mean different things depending on context. e.g. "MS" can mean multiple sclerosis, mitral stenosis or morphine sulfate. This program reads the words around an abbreviation and works out which meaning is the most likely. It shows the clue words behind its answer, and says "not sure" when the clues are weak.

Written in Python. Uses Naive Bayes.
There is some made-up sample data if you want to run it on that. This is how you do that:
```
python abbrev.py evaluate examples/fake_data.txt
python abbrev.py predict  examples/fake_data.txt MS "Echo shows MS with a narrow valve."
python test_abbrev.py
```

There is also a real dataset you can download to run on:
1. Download the anonymized data set from the [Clinical Abbreviation Sense Inventory](https://conservancy.umn.edu/items/6651323b-444a-479e-a41a-abca58c2e721) (University of Minnesota).
2. Put the file in a `data/` folder. It isn't included in this repository.
3. Run:
```
python abbrev.py evaluate data/YOUR_FILE.txt
python abbrev.py predict  data/YOUR_FILE.txt AB "Blood type is AB positive."
```

This project uses the Clinical Abbreviation Sense Inventory by Moon, Pakhomov & Melton (2012), University of Minnesota (https://hdl.handle.net/11299/137703), which is not included in this repository. 
Also, this project only recognizes the abbreviations and meanings in its training data. This is a learning project not a clinically validated tool.

## Data credit

Moon, S., Pakhomov, S., & Melton, G. (2012). *Clinical Abbreviation Sense Inventory*. University Digital Conservancy. https://hdl.handle.net/11299/137703

`examples/fake_data.txt` is entirely made up.
