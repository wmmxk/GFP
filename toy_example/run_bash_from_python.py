from lib.config import FILE_TEST
from lib.config import PATH_DATA
import subprocess


def test_bash_cmd():
    bash_cmd = "wc -l " + PATH_DATA+"/"+FILE_TEST +"| cut -d\" \" -f1"
    num_rows = subprocess.check_output(bash_cmd, shell=True).strip()
    num_rows = int(num_rows)
    print("res from bash: ", num_rows)

