.PHONY: clean
clean:
	find . -name __pycache__ | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf

.PHONY: style
style:
	isort --profile black src/ && black src/
