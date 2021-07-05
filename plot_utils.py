import matplotlib.pyplot as plt


def simple_plot(values, names, title, debug=False):
    fig = plt.figure(figsize=(16, 7))
    for v, n in zip(values, names):
        plt.plot(v, label=n)
    plt.legend(loc="upper left")
    plt.title(title)
    if debug:
        plt.show()
    return fig