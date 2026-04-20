.PHONY: install run test check

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

run:
	.venv/bin/streamlit run app.py

test:
	.venv/bin/python -m unittest discover -s tests

check:
	.venv/bin/python -m compileall app.py analytics.py webrtc_callbacks.py tests
	.venv/bin/python -m unittest discover -s tests
