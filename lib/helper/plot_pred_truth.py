import matplotlib.pyplot as plt
import numpy as np


def plot_pred_truth(truth, pred, fig_path, sanity=False):
    plt.scatter(truth, pred)
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    error = round(np.mean(np.absolute(truth - pred)), 2)
    plt.title("MAE: "+str(error))
    plt.savefig(fig_path, bbox_inches='tight')
    if sanity:
        plt.show()
