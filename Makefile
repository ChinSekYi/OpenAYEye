install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest test.py

format:
	isort $(shell find . -name "*.py")
	black $(shell find . -name "*.py")

run:
	python main.py

all: install format