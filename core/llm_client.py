"""
LLM Client for multi-provider integration (Groq, Anthropic, OpenAI, HuggingFace)
Implements free-tier optimization, fallback strategy, and confidence scoring.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging  # ✅ Use standard logging initially

# Create module-level logger that will be setup later
logger = logging.getLogger(__name__)

"""
LLM Client for multi-provider integration (Groq, Anthropic, OpenAI, HuggingFace)
Implements free-tier optimization, fallback strategy, and confidence scoring.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# Import free-tier providers
try:
    from groq import Groq
    from anthropic import Anthropic
    import openai
    from huggingface_hub import InferenceClient
except ImportError:
    # Graceful degradation for deployment
    pass

from utils.config import Config



class ModelProvider(Enum):
    """Supported LLM providers (free tiers only)"""
    GROQ = "groq"  # Llama 3.1 70B/8B
    ANTHROPIC = "anthropic"  # Claude 3.5 Haiku
    OPENAI = "openai"  # GPT-4o mini (free credits)
    HF = "huggingface"  # Llama 3.1 8B via HF Inference API
    GEMINI = "gemini"  # Gemini 1.5 Flash (free tier)


@dataclass
class LLMResponse:
    """Structured LLM response with confidence metrics"""
    content: str
    provider: ModelProvider
    model: str
    tokens_used: int
    processing_time: float
    confidence_score: float  # 0.0 to 1.0
    reasoning: Optional[str] = None
    should_abstain: bool = False  # If confidence < threshold


class LLMClient:
    """
    Multi-provider LLM client with intelligent fallback and confidence calibration.
    Optimized for free-tier usage with rate limiting and cost tracking.
    """

    def __init__(self):
        self.config = Config()
        self._init_clients()
        self.usage_stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "provider_usage": {p.value: 0 for p in ModelProvider}
        }

        # Provider priority based on task type
        self.provider_priority = {
            "reasoning": [ModelProvider.GROQ, ModelProvider.ANTHROPIC, ModelProvider.OPENAI],
            "creative": [ModelProvider.ANTHROPIC, ModelProvider.GROQ, ModelProvider.GEMINI],
            "coding": [ModelProvider.GROQ, ModelProvider.OPENAI, ModelProvider.HF],
            "analysis": [ModelProvider.ANTHROPIC, ModelProvider.GROQ, ModelProvider.OPENAI],
            "summarization": [ModelProvider.GEMINI, ModelProvider.HF, ModelProvider.GROQ]
        }

    def _init_clients(self):
        """Initialize API clients with environment variables"""
        self.clients = {}

        try:
            # Groq (Llama 3.1 70B - free tier)
            if os.getenv("GROQ_API_KEY"):
                self.clients[ModelProvider.GROQ] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("✓ Groq client initialized")
        except Exception as e:
            logger.warning(f"Groq initialization failed: {e}")

        try:
            # Anthropic (Claude 3.5 Haiku - free tier via trial)
            if os.getenv("ANTHROPIC_API_KEY"):
                self.clients[ModelProvider.ANTHROPIC] = Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("✓ Anthropic client initialized")
        except Exception as e:
            logger.warning(f"Anthropic initialization failed: {e}")

        try:
            # OpenAI (GPT-4o mini - free credits)
            if os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.clients[ModelProvider.OPENAI] = openai.OpenAI()
                logger.info("✓ OpenAI client initialized")
        except Exception as e:
            logger.warning(f"OpenAI initialization failed: {e}")

        try:
            # HuggingFace (Free Inference API)
            if os.getenv("HF_TOKEN"):
                self.clients[ModelProvider.HF] = InferenceClient(
                    token=os.getenv("HF_TOKEN")
                )
                logger.info("✓ HuggingFace client initialized")
        except Exception as e:
            logger.warning(f"HuggingFace initialization failed: {e}")

        if not self.clients:
            logger.error("❌ No LLM providers available. Check API keys.")
            raise RuntimeError("At least one LLM provider must be configured")

    def _calculate_confidence(self, response: str, task_type: str) -> float:
        """
        Calculate confidence score based on response characteristics.
        This is crucial for the uncertainty agent's decisions.
        """
        confidence = 0.7  # Base confidence

        # Heuristic confidence scoring
        indicators = {
            "admits_uncertainty": ["i don't know", "not sure", "uncertain", "cannot determine"],
            "high_confidence": ["certain", "definitely", "conclusively", "evidence shows"],
            "hedging": ["might", "could", "possibly", "perhaps", "likely"]
        }

        response_lower = response.lower()

        # Penalize uncertainty admissions
        for phrase in indicators["admits_uncertainty"]:
            if phrase in response_lower:
                confidence -= 0.3
                break

        # Boost for high confidence language
        for phrase in indicators["high_confidence"]:
            if phrase in response_lower:
                confidence += 0.15
                break

        # Mild penalty for excessive hedging
        hedge_count = sum(1 for phrase in indicators["hedging"] if phrase in response_lower)
        if hedge_count > 3:
            confidence -= hedge_count * 0.05

        # Length-based confidence (too short might be incomplete)
        if len(response.split()) < 10:
            confidence -= 0.2
        elif len(response.split()) > 200:
            confidence += 0.1  # Thorough responses

        # Clamp to 0-1 range
        return max(0.0, min(1.0, confidence))

    def _select_provider(self, task_type: str, complexity: str = "medium") -> ModelProvider:
        """
        Select optimal provider based on task and current usage.
        Implements load balancing across free tiers.
        """
        candidate_providers = self.provider_priority.get(task_type, list(ModelProvider))

        # Filter to available providers
        available = [p for p in candidate_providers if p in self.clients]

        if not available:
            # Fallback to any available provider
            available = list(self.clients.keys())

        if not available:
            raise RuntimeError("No LLM providers available")

        # Select provider with least usage (load balancing)
        selected = min(available, key=lambda p: self.usage_stats["provider_usage"][p.value])

        logger.debug(f"Selected provider {selected.value} for {task_type} task")
        return selected

    async def generate(
            self,
            prompt: str,
            task_type: str = "reasoning",
            temperature: float = 0.3,
            max_tokens: int = 2000,
            require_confidence: bool = True,
            **kwargs
    ) -> LLMResponse:
        """
        Generate a response with automatic provider selection and confidence scoring.

        Args:
            prompt: The input prompt
            task_type: Type of task (affects provider selection)
            temperature: Creativity control (0.0-1.0)
            max_tokens: Maximum response length
            require_confidence: If True, calculate confidence score

        Returns:
            LLMResponse object with content and metadata
        """
        start_time = datetime.now()
        self.usage_stats["total_requests"] += 1

        provider = self._select_provider(task_type)
        model_map = {
            ModelProvider.GROQ: "llama-3.1-70b-versatile",
            ModelProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
            ModelProvider.OPENAI: "gpt-4o-mini",
            ModelProvider.HF: "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ModelProvider.GEMINI: "gemini-1.5-flash"
        }

        model = model_map.get(provider, "unknown")

        try:
            response_content = ""
            tokens_used = 0

            if provider == ModelProvider.GROQ:
                response = self.clients[provider].chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

            elif provider == ModelProvider.ANTHROPIC:
                response = self.clients[provider].messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_content = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            elif provider == ModelProvider.OPENAI:
                response = self.clients[provider].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

            elif provider == ModelProvider.HF:
                response = self.clients[provider].text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                response_content = response
                tokens_used = len(response.split())  # Approximate

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Calculate confidence
            confidence = self._calculate_confidence(response_content, task_type) if require_confidence else 1.0

            # Update usage stats
            self.usage_stats["provider_usage"][provider.value] += 1
            self.usage_stats["tokens_used"] += tokens_used

            processing_time = (datetime.now() - start_time).total_seconds()

            # Determine if should abstain (based on confidence threshold)
            should_abstain = confidence < self.config.MIN_CONFIDENCE_THRESHOLD

            logger.info(f"LLM generation: {provider.value} | Confidence: {confidence:.2f} | Tokens: {tokens_used}")

            return LLMResponse(
                content=response_content,
                provider=provider,
                model=model,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence_score=confidence,
                should_abstain=should_abstain
            )

        except Exception as e:
            self.usage_stats["failed_requests"] += 1
            logger.error(f"LLM generation failed with {provider.value}: {e}")

            # Fallback to next provider if available
            if len(self.clients) > 1:
                logger.info(f"Attempting fallback for failed request...")
                # Remove failed provider temporarily
                fallback_providers = [p for p in self.clients.keys() if p != provider]
                if fallback_providers:
                    # Recursive call with next provider
                    kwargs["retry_count"] = kwargs.get("retry_count", 0) + 1
                    if kwargs["retry_count"] < len(self.clients):
                        return await self.generate(
                            prompt, task_type, temperature, max_tokens,
                            require_confidence, **kwargs
                        )

            # If all providers fail or max retries reached
            return LLMResponse(
                content=f"Error: Unable to generate response. All providers failed.",
                provider=provider,
                model=model,
                tokens_used=0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                confidence_score=0.0,
                should_abstain=True
            )

    async def generate_with_reasoning(
            self,
            prompt: str,
            task_type: str = "reasoning"
    ) -> Tuple[str, str, float]:
        """
        Generate a response with explicit reasoning chain (for critic/uncertainty agents).

        Returns:
            Tuple of (final_answer, reasoning_chain, confidence_score)
        """
        reasoning_prompt = f"""You are a research assistant. For the following task, provide:
1. Your step-by-step reasoning process
2. Your final answer
3. Your confidence level (0-100%)

Task: {prompt}

Format your response exactly as:
REASONING: [your reasoning here]
ANSWER: [your final answer here]
CONFIDENCE: [number between 0-100]"""

        response = await self.generate(
            reasoning_prompt,
            task_type=task_type,
            temperature=0.1,  # Low temperature for reasoning
            max_tokens=3000
        )

        # Parse the structured response
        content = response.content
        reasoning = ""
        answer = ""
        confidence = response.confidence_score * 100  # Convert to percentage

        try:
            lines = content.split('\n')
            current_section = None

            for line in lines:
                if line.startswith("REASONING:"):
                    current_section = "reasoning"
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("ANSWER:"):
                    current_section = "answer"
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_text = line.replace("CONFIDENCE:", "").strip()
                        confidence = float(conf_text) / 100.0  # Convert to 0-1
                    except:
                        pass
                    current_section = None
                elif current_section == "reasoning":
                    reasoning += "\n" + line.strip()
                elif current_section == "answer":
                    answer += "\n" + line.strip()

        except Exception as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            # Fallback: use entire response as answer
            answer = content

        return answer, reasoning, confidence

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            **self.usage_stats,
            "available_providers": [p.value for p in self.clients.keys()],
            "timestamp": datetime.now().isoformat()
        }

    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "provider_usage": {p.value: 0 for p in ModelProvider}
        }


# Global singleton instance
_llm_client_instance = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance"""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()
    return _llm_client_instance