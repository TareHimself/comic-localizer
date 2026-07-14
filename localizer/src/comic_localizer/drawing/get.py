from typing import Sequence
from comic_localizer.core.plugin import Drawer
from comic_localizer.drawing.horizontal import HorizontalDrawer

_drawing_data = list(
    filter(
        lambda a: a.is_valid(),
        [HorizontalDrawer],
    )
)


def get_drawers() -> Sequence[type[Drawer]]:
    return _drawing_data
