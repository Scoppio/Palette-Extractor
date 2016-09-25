init:
    pip install -r requirements.txt

test:
    py.test test_palette_extractor.py

.PHONY: init test