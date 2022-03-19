from os.path import splitext
import matplotlib.pyplot as plt

def plot_history(outpath, legend=[], plots=[]):
    fig_path = ''.join(splitext(outpath)[:-1] + ('.png',))
    for i in range(len(plots)):
        plt.subplot()
        plt.xlabel('Epoch')
        plt.ylabel(legend[i])
        plt.plot(range(len(plots[i])), plots[i])
    plt.savefig(fig_path)

