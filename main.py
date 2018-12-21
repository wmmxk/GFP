from lib.helper.count_num_rows import count_num_rows
from lib.train import Train
from lib.config import CHUNK_SIZE
from lib.helper.merge_mul_df import merge_mul_df


def main():
    num_rows = count_num_rows()
    num_iter = (num_rows+CHUNK_SIZE-1) // CHUNK_SIZE

    for index_chunk in range(num_iter):
        train = Train(index_chunk)
        train.cv()

    merge_mul_df()

main()
