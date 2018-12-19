import os
import pytest
from ..train import Train
from ..config import PATH_OUT_DATA


@pytest.fixture(scope="module")
def train():
    return Train()


def test_create_res_folder(train):
    cv_dir = os.path.join(PATH_OUT_DATA, "CV")
    assert os.path.isdir(cv_dir)



