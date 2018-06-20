## cython

Cythonize all the files

```sh
# python3 setup.py build_ext --inplace
echo "Cythonizing"
```

## test

Run all the tests

Run task `cython` before this

```sh
pytest
```

## test-debug

Run all the tests with `debug` logger.

Run task `cython` before this

```sh
pytest --log-cli-level debug
```
