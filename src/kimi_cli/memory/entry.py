"""The memory record and its kinds.

Re-exported from :mod:`carryover.entry`. The implementation lives in the Carryover
package — a workspace member under ``packages/carryover`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``carryover`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from carryover.entry import (
    BEHAVIOURAL_KINDS as BEHAVIOURAL_KINDS,
)
from carryover.entry import (
    LOOKUP_KINDS as LOOKUP_KINDS,
)
from carryover.entry import (
    MemoryEntry as MemoryEntry,
)
from carryover.entry import (
    MemoryKind as MemoryKind,
)
from carryover.entry import (
    MemoryScope as MemoryScope,
)

__all__ = [
    "BEHAVIOURAL_KINDS",
    "LOOKUP_KINDS",
    "MemoryEntry",
    "MemoryKind",
    "MemoryScope",
]
