from collections import defaultdict
from typing import Any, Dict, List


class ResultMerger:
    """
    Generic result merger.

    Supports:
    1. Legacy extractor format:
       {
           "fields": [
               {
                   "field": "...",
                   "value": "...",
                   "confidence": 0.9,
                   "evidence": "..."
               }
           ]
       }

    2. New generic extractor format:
       {
           "field": "...",
           "value": "...",
           "chunk_id": "...",
           "source_type": "..."
       }

    The merger is document-agnostic.
    """

    def merge(self, results):
        if not isinstance(results, list):
            return {}

        fields = defaultdict(list)

        for result in results:

            if not isinstance(result, dict):
                continue

            # -------------------------------------------------
            # FORMAT 1: legacy {"fields": [...]}
            # -------------------------------------------------
            if "fields" in result:

                items = result.get("fields", [])

                if not isinstance(items, list):
                    continue

                for item in items:

                    if not isinstance(item, dict):
                        continue

                    self._add_item(
                        fields,
                        item
                    )

            # -------------------------------------------------
            # FORMAT 2: generic flat result
            # -------------------------------------------------
            elif "field" in result:

                self._add_item(
                    fields,
                    result
                )

        return dict(fields)

    def _add_item(
        self,
        fields,
        item: Dict[str, Any]
    ):

        field = item.get("field")

        if field is None:
            return

        field = str(field).strip()

        if not field:
            return

        value = item.get("value")

        if value is None:
            return

        if isinstance(value, (dict, list)):
            value = str(value)

        value = str(value).strip()

        if not value:
            return

        # New extractor does not necessarily provide confidence.
        try:
            confidence = float(
                item.get(
                    "confidence",
                    1.0
                )
            )
        except (TypeError, ValueError):
            confidence = 1.0

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        # Support both old evidence and new source metadata.
        evidence = item.get(
            "evidence",
            ""
        )

        if not evidence:
            chunk_id = item.get(
                "chunk_id",
                ""
            )

            source_type = item.get(
                "source_type",
                ""
            )

            metadata = []

            if chunk_id:
                metadata.append(
                    f"chunk={chunk_id}"
                )

            if source_type:
                metadata.append(
                    f"source={source_type}"
                )

            evidence = ", ".join(metadata)

        fields[field].append(
            {
                "value": value,
                "confidence": confidence,
                "evidence": str(
                    evidence or ""
                ).strip(),
            }
        )


# -------------------------------------------------------------
# Backward-compatible function
# -------------------------------------------------------------

def merge_extractions(results):
    """
    Backward-compatible wrapper.

    Existing code that calls:

        merge_extractions(results)

    will continue to work.
    """

    merger = ResultMerger()

    return merger.merge(results)