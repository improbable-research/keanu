import keanu
import re

def test_version_is_correctly_formatted() -> None:
    version = keanu.__version__
    assert isinstance(version, str)
    # Example versions that pass: 0.0.18.dev1 0.0.18
    # Example versions that fail: 0.0.dev1 0.0
    pattern = re.compile(r'^(\d+\.)(\d+\.)(\d+)(\.(\w|-)+)?$')
    assert pattern.match(version)