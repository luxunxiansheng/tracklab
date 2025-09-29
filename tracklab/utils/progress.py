from typing import Iterable, Optional, Any

from tqdm import tqdm
from rich.progress import track

use_rich: bool = False


def progress(
    sequence: Iterable[Any], desc: str = "", total: Optional[int] = None
) -> Iterable[Any]:
    """Create a progress bar for an iterable sequence.

    Args:
        sequence: The iterable to track progress for.
        desc: Description for the progress bar.
        total: Total number of items (if known).

    Returns:
        An iterable with progress tracking.
    """
    if use_rich:
        return track(sequence, description=desc, total=total)
    else:
        return tqdm(sequence, desc=desc, total=total)
