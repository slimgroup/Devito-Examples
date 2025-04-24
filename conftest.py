import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector):
    if hasattr(collector, "fspath") and \
       str(collector.fspath).endswith(".ipynb"):
        collector.skip_compare += ('text/latex', 'stderr')
