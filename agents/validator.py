class ResultValidator:
    """
    Generic validation agent.

    Validates merged extraction results without assuming
    any document-specific fields or schema.
    """

    def __init__(self, confidence_threshold=0.60):
        self.confidence_threshold = confidence_threshold

    # =========================================================
    # MAIN VALIDATION
    # =========================================================

    def validate(self, extracted):

        if not isinstance(extracted, dict):
            return {
                "status": "REVIEW_REQUIRED",
                "issues": [
                    {
                        "issue": "Invalid extraction structure"
                    }
                ]
            }

        issues = []

        for field, values in extracted.items():

            # Normalize single value -> list
            if not isinstance(values, list):
                values = [values]

            for value in values:

                if not isinstance(value, dict):
                    issues.append({
                        "field": field,
                        "issue": "Invalid extracted field structure"
                    })
                    continue

                extracted_value = value.get("value")
                evidence = value.get("evidence", "")

                # -------------------------------------------------
                # Confidence
                # -------------------------------------------------

                try:
                    confidence = float(
                        value.get("confidence", 0.0)
                    )
                except (TypeError, ValueError):

                    confidence = 0.0

                    issues.append({
                        "field": field,
                        "issue": "Invalid confidence value"
                    })

                confidence = max(
                    0.0,
                    min(1.0, confidence)
                )

                # -------------------------------------------------
                # Missing value
                # -------------------------------------------------

                value_missing = (
                    extracted_value is None
                    or not str(extracted_value).strip()
                )

                if value_missing:

                    issues.append({
                        "field": field,
                        "issue": "Missing value",
                        "confidence": confidence,
                        "evidence": evidence,
                    })

                # -------------------------------------------------
                # Low confidence
                # -------------------------------------------------

                if confidence < self.confidence_threshold:

                    issues.append({
                        "field": field,
                        "issue": "Low confidence",
                        "confidence": confidence,
                        "evidence": evidence,
                    })

                # -------------------------------------------------
                # Missing evidence
                # -------------------------------------------------

                if (
                    not value_missing
                    and not str(evidence).strip()
                ):

                    issues.append({
                        "field": field,
                        "issue": "Missing evidence",
                        "confidence": confidence,
                    })

        return {
            "status": (
                "REVIEW_REQUIRED"
                if issues
                else "PASS"
            ),
            "issues": issues,
        }


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

class ValidationAgent(ResultValidator):
    """
    Backward-compatible alias for existing code.
    """
    pass


# =============================================================
# DETERMINISTIC NUMERICAL VALIDATION
# =============================================================

def validate_total(
    subtotal,
    tax,
    total,
    tolerance=0.01
):
    """
    Verify:

        subtotal + tax = total

    Returns True when the values match within
    the supplied tolerance.
    """

    try:

        expected = (
            float(subtotal)
            + float(tax)
        )

        actual = float(total)

        return abs(
            expected - actual
        ) <= tolerance

    except (TypeError, ValueError):

        return False