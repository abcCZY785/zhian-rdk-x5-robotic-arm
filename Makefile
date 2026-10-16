.PHONY: help setup demo vision test

help:
	@echo "Zhian RDK X5 Robotic Arm"
	@echo "Available commands:"
	@echo "  make setup   - install dependencies"
	@echo "  make demo    - run demo pipeline"
	@echo "  make vision  - run vision module"
	@echo "  make test    - run tests"

setup:
	@echo "TODO: install dependencies"

demo:
	@echo "TODO: run demo pipeline"

vision:
	@echo "TODO: run vision module"

test:
	python -m unittest discover -s tests -v
