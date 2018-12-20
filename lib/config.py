from collections import namedtuple

FILE_TRAIN = "GFP_lineup.csv"
FILE_TEST = "to_screen.csv"
CHUNK_SIZE = 100000  # predict on a chunk of the testing set

PATH_DATA_HOME = "/home/wxk/Data/GFP"
PATH_DATA = PATH_DATA_HOME + "/data"
PATH_OUT_DATA = PATH_DATA_HOME + "/out_data"

PATH_FIG = PATH_DATA_HOME + "/out_fig"
PATH_MODEL = PATH_DATA_HOME + "/model"

Config = namedtuple('Config', 'type_model, num_fold sanity')
CONFIG = Config(type_model='gp', num_fold=10, sanity=False)
