def select_highest_voted_att(atts, atts_confidences=None):
    confidence_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences

    # Iterate through the predictions to calculate the total confidence for each attribute
    for value, conf in zip(atts, atts_confidences):
        if value not in confidence_sum:
            confidence_sum[value] = 0
        confidence_sum[value] += conf

    if not confidence_sum:
        return None

    def _is_valid(val):
        if val is None:
            return False
        if isinstance(val, str):
            cleaned = val.strip()
            return cleaned != "" and cleaned.lower() not in {"none", "nan"}
        return True

    valid_candidates = [value for value in confidence_sum if _is_valid(value)]

    if valid_candidates:
        return max(valid_candidates, key=lambda v: confidence_sum[v])

    # If everything was empty/None, return the best-scoring entry as fallback
    return max(confidence_sum, key=lambda v: confidence_sum[v])
