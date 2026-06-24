import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer


def pytest_addoption(parser):
    parser.addoption(
        "--linearfold",
        default=None,
        help="Path to the LinearFold executable. Required to run linearfold integration tests.",
    )


@pytest.fixture(scope="session")
def linearfold_exe(request):
    path = request.config.getoption("--linearfold")
    if path is None:
        pytest.skip("Pass --linearfold /path/to/linearfold to run this test")
    return path


REPO_ROOT    = Path(__file__).resolve().parent.parent
LF_TEST_DATA = REPO_ROOT / "tests/linearfold_test_data"
MODEL_MAX_LENGTH = 1024

sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "YYLY66/mRNABERT"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, trust_remote_code=True,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="right",
    )


@pytest.fixture(scope="session")
def vocab(tokenizer):
    return tokenizer.get_vocab()


@pytest.fixture(scope="session")
def wc_full(tokenizer):
    from bias.wc import build_wc_lookup
    return build_wc_lookup(tokenizer, utr_only=False)


@pytest.fixture(scope="session")
def wc_utr_only(tokenizer):
    from bias.wc import build_wc_lookup
    return build_wc_lookup(tokenizer, utr_only=True)
