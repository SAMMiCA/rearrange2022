from typing import Sequence
import sys
import platform
import os
import glob
import Xlib
import pdb


def get_open_x_displays(throw_error_if_empty: bool = False) -> Sequence[str]:
    assert platform.system() == "Linux", "Can only get X-displays for Linux systems."

    displays = []

    open_display_strs = [
        os.path.basename(s)[1:] for s in glob.glob("/tmp/.X11-unix/X*")
    ]

    for open_display_str in sorted(open_display_strs):
        try:
            open_display_str = str(int(open_display_str))
            display = Xlib.display.Display(":{}".format(open_display_str))
        except Exception:
            continue

        displays.extend(
            [f"{open_display_str}.{i}" for i in range(display.screen_count())]
        )

    if throw_error_if_empty and len(displays) == 0:
        raise IOError(
            "Could not find any open X-displays on which to run AI2-THOR processes. "
            " Please see the AI2-THOR installation instructions at"
            " https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
            " for information as to how to start such displays."
        )

    return displays


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
