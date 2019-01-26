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

## test-coverage

```sh
py.test --cov-report term-missing --cov=bbox tests/ 
```

## build

```sh
rm -rf dist/*
python setup.py sdist bdist_wheel
```

## publish

Run task `build` before this

```sh
echo "publishing"
twine upload dist/*
```