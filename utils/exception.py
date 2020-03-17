import sys
from bdb import BdbQuit

__all__ = [
    "register_ipython_excepthook",
]


def register_ipython_excepthook() -> None:
    r"""Register an exception hook that launches an interactive IPython session upon uncaught exceptions.
    """

    def excepthook(type, value, traceback):
        if type in [KeyboardInterrupt, BdbQuit]:
            # Don't capture keyboard interrupts (Ctrl+C) or Python debugger exit.
            sys.__excepthook__(type, value, traceback)
        else:
            ipython_hook(type, value, traceback)

    # Enter IPython debugger on exception.
    from IPython.core import ultratb

    ipython_hook = ultratb.FormattedTB(mode='Context', color_scheme='Linux', call_pdb=1)
    sys.excepthook = excepthook
