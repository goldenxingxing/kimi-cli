"""Keeping the injected set affordable: supersession, dormancy, pressure.

Re-exported from :mod:`carryover.consolidate`. The implementation lives in the Carryover
package — a workspace member under ``packages/carryover`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``carryover`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from carryover.consolidate import (
    BEHAVIOURAL_BUDGET_CHARS as BEHAVIOURAL_BUDGET_CHARS,
)
from carryover.consolidate import (
    DORMANT_AFTER_DAYS as DORMANT_AFTER_DAYS,
)
from carryover.consolidate import (
    LONG_ENTRY_CHARS as LONG_ENTRY_CHARS,
)
from carryover.consolidate import (
    MAX_PROPOSALS as MAX_PROPOSALS,
)
from carryover.consolidate import (
    PRESSURE_ACT_AT as PRESSURE_ACT_AT,
)
from carryover.consolidate import (
    PRESSURE_WARN_AT as PRESSURE_WARN_AT,
)
from carryover.consolidate import (
    SUPERSEDED_RATIO as SUPERSEDED_RATIO,
)
from carryover.consolidate import (
    TOPIC_OVERLAP_RATIO as TOPIC_OVERLAP_RATIO,
)
from carryover.consolidate import (
    Supersession as Supersession,
)
from carryover.consolidate import (
    announces_supersession as announces_supersession,
)
from carryover.consolidate import (
    find_dormant as find_dormant,
)
from carryover.consolidate import (
    find_superseded as find_superseded,
)
from carryover.consolidate import (
    mark_relevant as mark_relevant,
)
from carryover.consolidate import (
    pressure as pressure,
)
from carryover.consolidate import (
    topic_terms as topic_terms,
)

__all__ = [
    "BEHAVIOURAL_BUDGET_CHARS",
    "DORMANT_AFTER_DAYS",
    "LONG_ENTRY_CHARS",
    "MAX_PROPOSALS",
    "PRESSURE_ACT_AT",
    "PRESSURE_WARN_AT",
    "SUPERSEDED_RATIO",
    "Supersession",
    "TOPIC_OVERLAP_RATIO",
    "announces_supersession",
    "find_dormant",
    "find_superseded",
    "mark_relevant",
    "pressure",
    "topic_terms",
]
