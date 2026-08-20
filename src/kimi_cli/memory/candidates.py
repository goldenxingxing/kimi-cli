"""Proposals awaiting a person's decision.

Re-exported from :mod:`amem.candidates`. The implementation lives in the Amem
package — a workspace member under ``packages/amem`` — because none of it is
specific to this application, and it was kept in step here only by copying.
That copying had already cost something: a hand-carried ``fold_text`` lost two
of its three normalisation axes and nothing noticed until a test was written.

Kept as a module rather than importing ``amem`` at each call site, so that
:mod:`kimi_cli.memory` stays the one seam between this application and whatever
provides its memory. Changing the provider is an edit in this package.
"""

from amem.candidates import (
    CANDIDATE_TTL_SECONDS as CANDIDATE_TTL_SECONDS,
)
from amem.candidates import (
    CANDIDATES_FILENAME as CANDIDATES_FILENAME,
)
from amem.candidates import (
    MAX_CANDIDATES as MAX_CANDIDATES,
)
from amem.candidates import (
    CandidateFile as CandidateFile,
)
from amem.candidates import (
    MemoryCandidate as MemoryCandidate,
)

__all__ = [
    "CANDIDATES_FILENAME",
    "CANDIDATE_TTL_SECONDS",
    "CandidateFile",
    "MAX_CANDIDATES",
    "MemoryCandidate",
]
