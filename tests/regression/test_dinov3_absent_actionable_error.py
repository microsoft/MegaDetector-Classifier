"""Phase 5 regression I13: absent DINOv3 deps → actionable ImportError.

Per plan §4.5 / binding directive #7: in an env without transformers,
`from src.models import classification` succeeds, AND
`classification.DINOv3CustomWeights` raises ImportError whose message
contains 'pip install -r requirements-dinov3.txt'.
"""
import subprocess
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    from src.models import classification
    try:
        _ = classification.DINOv3CustomWeights
    except ImportError as e:
        assert "pip install -r requirements-dinov3.txt" in str(e), str(e)
        print("OK")
    else:
        raise AssertionError("expected ImportError, got none")
""")


@pytest.mark.isolated_env
def test_absent_dinov3_raises_actionable_import_error():
    result = subprocess.run(
        ["uv", "run", "--isolated", "--with-requirements", "requirements.txt",
         "python", "-c", SCRIPT],
        capture_output=True, text=True, timeout=900, cwd="."
    )
    assert result.returncode == 0, (
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "OK" in result.stdout
