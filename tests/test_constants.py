import os

from constants import ROOT_DIR


def test_root_dir():
    assert ROOT_DIR == os.path.abspath(os.path.dirname(__file__)).split("\\tests")[0]
