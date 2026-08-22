"""Shortening a session summary to what stays useful.

Re-exported from :mod:`carryover.condense`. The implementation lives in the Carryover
package — a workspace member under ``packages/carryover`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``carryover`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from carryover.condense import (
    CROSS_SESSION_SECTIONS as CROSS_SESSION_SECTIONS,
)
from carryover.condense import (
    DEFAULT_SUMMARY_BUDGET as DEFAULT_SUMMARY_BUDGET,
)
from carryover.condense import (
    DROPPED_SECTIONS as DROPPED_SECTIONS,
)
from carryover.condense import (
    condense_summary as condense_summary,
)

__all__ = [
    "CROSS_SESSION_SECTIONS",
    "DEFAULT_SUMMARY_BUDGET",
    "DROPPED_SECTIONS",
    "condense_summary",
]
