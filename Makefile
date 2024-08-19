.PHONY: clean build

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type f -name "*.pyo" -exec rm -f {} +