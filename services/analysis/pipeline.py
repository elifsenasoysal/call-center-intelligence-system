def get_analysis(stt_output: dict) -> dict:
    """
    Analyzes a transcription result and returns sentiment, category, and summary.

    Args:
        stt_output: dict matching the STTOutput schema (see shared/contracts.py).

    Returns:
        dict with keys: overall_sentiment, sentiment_score, complaint_category,
        keywords, summary, agent_performance_score.
        See shared/mock_data/sample_llm_output.json for reference.
    """
    raise NotImplementedError
