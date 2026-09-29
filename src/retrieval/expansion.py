from typing import Any


def expand_neighbors(
    top_chunks: list[dict[str, Any]], vectorstore
) -> list[dict[str, Any]]:
    """Expands chunks with their previous and next neighbors on the same page."""
    needed_ids = set()
    for chunk in top_chunks:
        doc_id = chunk["metadata"]["doc_id"]
        page = chunk["metadata"]["page"]
        idx = chunk["metadata"]["chunk_index"]

        # prev
        if idx > 0:
            needed_ids.add(f"{doc_id}:{page}:{idx - 1}")
        # next
        needed_ids.add(f"{doc_id}:{page}:{idx + 1}")

    if not needed_ids:
        return top_chunks

    neighbors = vectorstore.get_by_ids(list(needed_ids))

    # Filter neighbors to only same page and add them
    expanded = {c["id"]: c for c in top_chunks}

    for n in neighbors:
        cid = n["id"]
        if cid not in expanded:
            n["is_neighbor_expansion"] = True
            n["relevance_band"] = "unset"
            expanded[cid] = n

    # Sort by doc, page, index
    result = list(expanded.values())
    result.sort(
        key=lambda x: (
            x["metadata"]["doc_id"],
            x["metadata"]["page"],
            x["metadata"]["chunk_index"],
        )
    )
    return result
