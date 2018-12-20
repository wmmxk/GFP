import subprocess

from ..config import FILE_TEST
from ..config import PATH_DATA


def count_num_rows():
    """
    count the number rows in the FILE_TEST
    :return: int
    """
    bash_cmd = "wc -l " + PATH_DATA+"/"+FILE_TEST + "| cut -d\" \" -f1"
    num_rows = subprocess.check_output(bash_cmd, shell=True).strip()
    return int(num_rows)

