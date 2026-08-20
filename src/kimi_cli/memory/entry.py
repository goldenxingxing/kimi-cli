"""The memory record and its kinds.

Re-exported from :mod:`amem.entry`. The implementation lives in the Amem
package — a workspace member under ``packages/amem`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``amem`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from amem.entry import (
    BEHAVIOURAL_KINDS as BEHAVIOURAL_KINDS,
)
from amem.entry import (
    LOOKUP_KINDS as LOOKUP_KINDS,
)
from amem.entry import (
    MemoryEntry as MemoryEntry,
)
from amem.entry import (
    MemoryKind as MemoryKind,
)
from amem.entry import (
    MemoryScope as MemoryScope,
)

__all__ = [
    "BEHAVIOURAL_KINDS",
    "LOOKUP_KINDS",
    "MemoryEntry",
    "MemoryKind",
    "MemoryScope",
]
