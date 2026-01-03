"""
Safe LLM Response Wrapper
"""
import json
import re

def safe_parse_llm_response(response):
    """
    Safely parse LLM response which could be dict or have attributes

    Args:
        response: LLM response (dict or object)

    Returns:
        tuple: (content_dict_or_str, confidence_score)
    """
    if isinstance(response, dict):
        content = response.get("content", "")
        confidence = response.get("confidence_score", 0.7)
    else:
        # Try attribute access
        try:
            content = response.content
            confidence = response.confidence_score
        except AttributeError:
            content = str(response)
            confidence = 0.7

    # Try to parse as JSON
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
            return parsed, confidence
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed, confidence
                except:
                    pass

    return content, confidence

def safe_json_loads(content, default=None):
    """Safely parse JSON with fallback"""
    if default is None:
        default = {}

    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        return default

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return default
