"""Recent session summaries.

Re-exported from :mod:`amem.recent`. The implementation lives in the Amem
package — a workspace member under ``packages/amem`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``amem`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from amem.recent import (
    DEFAULT_MAX_SUMMARIES as DEFAULT_MAX_SUMMARIES,
)
from amem.recent import (
    RECENT_FILENAME as RECENT_FILENAME,
)
from amem.recent import (
    SessionSummary as SessionSummary,
)
from amem.recent import (
    SummaryAppendResult as SummaryAppendResult,
)
from amem.recent import (
    SummaryTrigger as SummaryTrigger,
)
from amem.recent import (
    append_summary as append_summary,
)
from amem.recent import (
    read_recent_summaries as read_recent_summaries,
)
from amem.recent import (
    trim_old_summaries as trim_old_summaries,
)

__all__ = [
    "DEFAULT_MAX_SUMMARIES",
    "RECENT_FILENAME",
    "SessionSummary",
    "SummaryAppendResult",
    "SummaryTrigger",
    "append_summary",
    "read_recent_summaries",
    "trim_old_summaries",
]
