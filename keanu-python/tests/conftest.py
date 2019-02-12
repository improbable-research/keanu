import pytest

from keanu import set_deterministic_state

@pytest.fixture(autouse = True)
def make_tests_deterministic(request):
    set_deterministic_state()
