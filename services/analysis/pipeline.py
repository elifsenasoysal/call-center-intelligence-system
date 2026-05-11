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
    import json
    import os

    mock_path = os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'mock_data', 'sample_llm_output.json')
    try:
        with open(mock_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "complaint_category": "unknown",
            "keywords": ["mock", "data"],
            "summary": "This is a mock summary because the actual LLM integration is pending.",
            "agent_performance_score": 0.5
        }
