"""
Base Agent class with LLM integration
"""
import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.confidence = 0.5
        self.min_confidence = 0.6

        # Initialize LLM client
        try:
            from utils.llm_client import llm_client
            self.llm = llm_client
        except ImportError as e:
            logger.warning(f"LLM client not available: {e}")
            self.llm = None

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent logic - override in subclasses"""
        raise NotImplementedError("Subclasses must implement execute()")

    async def llm_generate(self,
                          prompt: str,
                          task_type: str = "creative",
                          max_tokens: int = 2000,
                          temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using LLM and return consistent response"""
        try:
            # Try to use the LLM client
            if hasattr(self, 'llm') and self.llm:
                result = await self.llm.generate(
                    prompt=prompt,
                    task_type=task_type,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # Ensure consistent response format
                if isinstance(result, dict):
                    return result
                else:
                    # Convert object to dict if needed
                    return {
                        "content": getattr(result, 'content', str(result)),
                        "confidence_score": getattr(result, 'confidence_score', 0.5)
                    }
            else:
                # Mock response for testing
                mock_response = {
                    "questions": [
                        {
                            "question": f"Mock research question about {task_type}",
                            "novelty_score": 8,
                            "feasibility": "high",
                            "synthesis_required": ["concept1", "concept2"],
                            "synthesis_explanation": "Mock explanation",
                            "data_requirements": ["data1", "data2"],
                            "experiment_approaches": ["approach1"],
                            "potential_impact": "Mock impact"
                        }
                    ],
                    "generation_rationale": "Mock rationale for testing"
                }

                return {
                    "content": json.dumps(mock_response),
                    "confidence_score": 0.7,
                    "success": True
                }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "content": json.dumps({"error": str(e)}),
                "confidence_score": 0.1,
                "success": False
            }

    def format_response(self,
                       success: bool,
                       data: Dict[str, Any],
                       message: str = "") -> Dict[str, Any]:
        """Format agent response consistently"""
        return {
            "success": success,
            "confidence": self.confidence,
            "should_abstain": self.confidence < self.min_confidence,
            "data": data,
            "message": message,
            "agent_name": self.name,
            "agent_description": self.description
        }

    def update_confidence(self, confidence: float):
        """Update agent confidence"""
        self.confidence = max(0.0, min(1.0, confidence))

    def calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score from metrics"""
        if not metrics:
            return 0.5

        weights = {
            "completeness": 0.3,
            "data_quality": 0.3,
            "validation_passed": 0.2,
            "design_quality": 0.2,
            "reliability": 0.2
        }

        total_weight = 0
        weighted_sum = 0

        for metric, value in metrics.items():
            weight = weights.get(metric, 0.1)
            weighted_sum += value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5


    def safe_parse_llm_response(self, response):
        """Safely parse LLM response which could be dict or have attributes"""
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
        try:
            parsed = json.loads(content)
            return parsed, confidence
        except json.JSONDecodeError:
            # Return as text
            return content, confidence

    async def store_in_memory(self, content: str, metadata: Dict[str, Any] = None):
        """Store information in memory"""
        # This would connect to your vector memory
        logger.info(f"Would store in memory: {content[:50]}...")
        pass