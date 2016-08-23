# spyctra

Helper tools for analyzing Raman and other spectroscopy data.

## Installation

Cloning the repo
```shell
git clone git@github.com:parkin/spyctra.git
cd spyctra
pip install .
```

Using pip
```shell
$ pip install git+git://github.com/parkin/spyctra.git#egg=spyctra
```

## Running tests

Using [nose](http://nose.readthedocs.io/en/latest/).
```shell
nosetests
```

## Documentation

Please see the [source code](spyctra) and corresponding [tests](tests) for further example usage.

### Baseline fitting

[`spyctra.baseline`](spyctra/baseline.py) implements baseline correction using asymmetrically
reweighted penalized least squares smoothing [arPLS](http://pubs.rsc.org/en/Content/ArticleLanding/2015/AN/C4AN01061B#!divAbstract).

to use

```python
from spyctra import arPLS
# y is some 1D spectrum
# z will be the baseline of y
z = arPLS(y)
```
