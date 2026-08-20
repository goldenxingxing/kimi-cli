"""Retrieval over stored memory.

Re-exported from :mod:`amem.search`. The implementation lives in the Amem
package — a workspace member under ``packages/amem`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``amem`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from amem.search import (
    FTS_MIN_QUERY_LEN as FTS_MIN_QUERY_LEN,
)
from amem.search import (
    MemorySearchIndex as MemorySearchIndex,
)
from amem.search import (
    SearchHit as SearchHit,
)

__all__ = [
    "FTS_MIN_QUERY_LEN",
    "MemorySearchIndex",
    "SearchHit",
]
