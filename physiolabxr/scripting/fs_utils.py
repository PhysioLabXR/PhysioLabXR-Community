from datetime import datetime


def get_datetime_str():
    return datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

def show_figure(figure, title, out_dir=None):
    """
    Show a matplotlib figure
    :param figure: matplotlib figure
    """
    if out_dir is not None:
        figure.savefig(f'{out_dir}/{title}.png')
    else:
        figure.show()