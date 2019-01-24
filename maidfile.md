## test

Run all the tests

```sh
pytest
```

## test-debug

Run all the tests with `debug` logger.

```sh
pytest --log-cli-level debug
```

## build

```sh
rm -rf dist/*
python setup.py sdist bdist_wheel
```

## publish

```sh
maid build
twine upload dist/*
```