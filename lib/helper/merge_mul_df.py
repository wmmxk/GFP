import pandas as pd
from ..config import PATH_OUT_DATA
from glob import glob
import os
from ..config import CONFIG


def merge_mul_df():
    """
    merge the prediction on each chunk of the X_test
    :return:
    """
    data_dir = os.path.join(PATH_OUT_DATA, "pred")
    files = glob(os.path.join(data_dir, CONFIG.type_model+"*chunk*"))
    files.sort(key=lambda x: int(x.split("_")[-1][:-4]))
    dfs = []
    for file in files:
       df = pd.read_csv(file)
       dfs.append(df)
    res = pd.concat(dfs, sort=False)
    res.sort_values(by=res.columns[-1], ascending=False, inplace=True)
    res.to_csv(os.path.join(data_dir, CONFIG.type_model+"pred_all.csv"), index=False)


