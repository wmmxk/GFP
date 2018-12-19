import os


def create_dir(my_dir):
    try:
        os.stat(my_dir)
    except:
        os.mkdir(my_dir)
