"""Reading and writing the store file.

Re-exported from :mod:`carryover.storage`. The implementation lives in the Carryover
package — a workspace member under ``packages/carryover`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``carryover`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from carryover.storage import (
    PERSISTENT_FILENAME as PERSISTENT_FILENAME,
)
from carryover.storage import (
    AmbiguousHandleError as AmbiguousHandleError,
)
from carryover.storage import (
    UpsertResult as UpsertResult,
)
from carryover.storage import (
    delete_entry as delete_entry,
)
from carryover.storage import (
    read_entries as read_entries,
)
from carryover.storage import (
    resolve_handle as resolve_handle,
)
from carryover.storage import (
    set_affirmed as set_affirmed,
)
from carryover.storage import (
    set_retired as set_retired,
)
from carryover.storage import (
    stamp_relevance as stamp_relevance,
)
from carryover.storage import (
    update_entry as update_entry,
)
from carryover.storage import (
    upsert_entry as upsert_entry,
)

__all__ = [
    "AmbiguousHandleError",
    "PERSISTENT_FILENAME",
    "UpsertResult",
    "delete_entry",
    "read_entries",
    "resolve_handle",
    "set_affirmed",
    "set_retired",
    "stamp_relevance",
    "update_entry",
    "upsert_entry",
]
