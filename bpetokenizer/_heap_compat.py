"""Max-heap compatibility shim.

``heapq.heapify_max``, ``heapq.heappop_max``, and ``heapq.heappush_max`` were
added as public API in Python 3.13.  On Python 3.12 we emulate them by
negating the priority value so a standard min-heap behaves like a max-heap.

All three functions operate on heaps whose entries are ``(priority, payload)``
tuples.  The shim is fully transparent: callers always see positive priorities.
"""

from __future__ import annotations

import heapq
import sys

if sys.version_info >= (3, 14):
    from heapq import heapify_max, heappop_max, heappush_max  # type: ignore[attr-defined]
else:

    def heapify_max(heap: list) -> None:  # type: ignore[misc]
        """Transform *heap* into a max-heap in-place."""
        for i in range(len(heap)):
            v, k = heap[i]
            heap[i] = (-v, k)
        heapq.heapify(heap)

    def heappop_max(heap: list) -> tuple:  # type: ignore[misc]
        """Pop and return the largest item from *heap*."""
        neg_v, k = heapq.heappop(heap)
        return (-neg_v, k)

    def heappush_max(heap: list, item: tuple) -> None:  # type: ignore[misc]
        """Push *item* onto *heap*, maintaining the max-heap invariant."""
        v, k = item
        heapq.heappush(heap, (-v, k))


__all__ = ["heapify_max", "heappop_max", "heappush_max"]
