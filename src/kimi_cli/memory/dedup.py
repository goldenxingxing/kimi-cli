"""Deciding whether a new entry restates one already held.

Re-exported from :mod:`amem.dedup`. The implementation lives in the Amem
package — a workspace member under ``packages/amem`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``amem`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from amem.dedup import (
    ADVISORY_RATIO as ADVISORY_RATIO,
)
from amem.dedup import (
    AUTO_MERGE_RATIO as AUTO_MERGE_RATIO,
)
from amem.dedup import (
    CHAR_LEVEL_MAX_LEN as CHAR_LEVEL_MAX_LEN,
)
from amem.dedup import (
    CREATE_VERDICT as CREATE_VERDICT,
)
from amem.dedup import (
    CROSS_SESSION_JACCARD as CROSS_SESSION_JACCARD,
)
from amem.dedup import (
    MAX_ADVISORIES as MAX_ADVISORIES,
)
from amem.dedup import (
    MIN_LENGTH_RATIO as MIN_LENGTH_RATIO,
)
from amem.dedup import (
    DuplicateAction as DuplicateAction,
)
from amem.dedup import (
    DuplicateVerdict as DuplicateVerdict,
)
from amem.dedup import (
    SummaryPlacement as SummaryPlacement,
)
from amem.dedup import (
    SummaryPolicy as SummaryPolicy,
)
from amem.dedup import (
    classify_entry as classify_entry,
)
from amem.dedup import (
    compact_entries as compact_entries,
)
from amem.dedup import (
    compact_summaries as compact_summaries,
)
from amem.dedup import (
    has_negation as has_negation,
)
from amem.dedup import (
    may_merge as may_merge,
)
from amem.dedup import (
    merge_entry as merge_entry,
)
from amem.dedup import (
    normalize_content as normalize_content,
)
from amem.dedup import (
    numeric_tokens as numeric_tokens,
)
from amem.dedup import (
    place_summary as place_summary,
)
from amem.dedup import (
    raw_similarity as raw_similarity,
)
from amem.dedup import (
    similarity as similarity,
)

__all__ = [
    "ADVISORY_RATIO",
    "AUTO_MERGE_RATIO",
    "CHAR_LEVEL_MAX_LEN",
    "CREATE_VERDICT",
    "CROSS_SESSION_JACCARD",
    "DuplicateAction",
    "DuplicateVerdict",
    "MAX_ADVISORIES",
    "MIN_LENGTH_RATIO",
    "SummaryPlacement",
    "SummaryPolicy",
    "classify_entry",
    "compact_entries",
    "compact_summaries",
    "has_negation",
    "may_merge",
    "merge_entry",
    "normalize_content",
    "numeric_tokens",
    "place_summary",
    "raw_similarity",
    "similarity",
]
