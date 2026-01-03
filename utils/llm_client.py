"""
LLM Client for various providers
"""
import json
import asyncio
from typing import Dict, Any
from utils.config import Config


class LLMClient:
    def __init__(self):
        self.providers = {}
        self._setup_providers()

    def _setup_providers(self):
        """Setup available LLM providers"""
        # For free-tier, we'll use mock provider and try Groq if available
        try:
            if Config.GROQ_API_KEY:
                self.providers['groq'] = GroqProvider()
                print("✅ Groq provider available")
        except:
            pass

        # Always have mock provider as fallback
        self.providers['mock'] = MockProvider()
        print("✅ Mock provider available (fallback)")

    async def generate(self, prompt: str, task_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Generate text using available LLM"""
        # Try real providers first
        for provider_name, provider in self.providers.items():
            if provider_name != 'mock':  # Try real providers first
                try:
                    print(f"🔄 Trying {provider_name} provider...")
                    result = await provider.generate(prompt, task_type, **kwargs)
                    if result.get("success", True):
                        print(f"✅ Using {provider_name} provider")
                        return result
                except Exception as e:
                    print(f"⚠️ {provider_name} provider failed: {e}")
                    continue

        # Fallback to mock
        print("🔄 Falling back to mock provider")
        return await self.providers['mock'].generate(prompt, task_type, **kwargs)


class BaseProvider:
    async def generate(self, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class MockProvider(BaseProvider):
    async def generate(self, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """Mock LLM response for testing"""
        await asyncio.sleep(0.1)  # Simulate processing

        # Different responses based on task type and prompt
        prompt_lower = prompt.lower()

        # Domain scouting response
        if "domain" in prompt_lower or "scout" in prompt_lower or "emerging" in prompt_lower:
            mock_response = {
                "domains": [
                    {
                        "name": "Neuro-Symbolic AI Integration",
                        "description": "Combining neural networks with symbolic reasoning for explainable AI",
                        "evidence": ["arXiv papers 2024", "GitHub repos trending", "2024 conferences focus"],
                        "trend_score": 85,
                        "novelty": "high",
                        "confidence": 0.8,
                        "data_availability": "high"
                    },
                    {
                        "name": "Quantum Machine Learning for Drug Discovery",
                        "description": "Applying quantum computing principles to accelerate pharmaceutical research",
                        "evidence": ["Research papers 2024", "Patents filed", "Industry reports"],
                        "trend_score": 80,
                        "novelty": "high",
                        "confidence": 0.75,
                        "data_availability": "medium"
                    },
                    {
                        "name": "AI-Assisted Synthetic Biology",
                        "description": "Using AI to design and optimize biological systems and synthetic organisms",
                        "evidence": ["Nature publications", "Startup funding", "Lab automation"],
                        "trend_score": 78,
                        "novelty": "high",
                        "confidence": 0.7,
                        "data_availability": "medium"
                    }
                ],
                "analysis": "These domains show high growth potential in 2024 with available research data",
                "search_queries_used": ["new arXiv categories 2024", "rising GitHub repos scientific 2024", "patent trends 2024"]
            }

        # Question generation response
        elif "question" in prompt_lower or "generate" in prompt_lower or "research question" in prompt_lower:
            mock_response = {
                "questions": [
                    {
                        "question": "How can neuro-symbolic integration improve the interpretability of deep learning models in healthcare diagnostics while maintaining state-of-the-art accuracy?",
                        "novelty_score": 8,
                        "feasibility": "high",
                        "synthesis_required": ["neural networks", "symbolic reasoning", "healthcare diagnostics"],
                        "synthesis_explanation": "Combines neural network pattern recognition with symbolic logic for explainable medical AI, requiring synthesis of computer vision and knowledge representation",
                        "data_requirements": ["medical imaging datasets", "clinical notes", "expert annotations", "knowledge graphs"],
                        "experiment_approaches": ["comparative study of accuracy vs interpretability", "ablation study of symbolic components", "clinical validation with radiologists"],
                        "potential_impact": "Could enable trustworthy AI diagnostics with human-interpretable reasoning chains"
                    },
                    {
                        "question": "What is the optimal trade-off between quantum circuit depth and classical neural network complexity for predicting molecular binding affinities in drug discovery?",
                        "novelty_score": 9,
                        "feasibility": "medium",
                        "synthesis_required": ["quantum computing", "deep learning", "computational chemistry"],
                        "synthesis_explanation": "Requires synthesizing quantum algorithms with classical ML and cheminformatics to create hybrid quantum-classical models",
                        "data_requirements": ["molecular databases", "quantum simulation results", "experimental binding data"],
                        "experiment_approaches": ["systematic architecture search", "benchmark on standardized datasets", "scalability analysis"],
                        "potential_impact": "Could accelerate drug discovery by orders of magnitude through quantum advantage"
                    }
                ],
                "generation_rationale": "Questions synthesized from emerging trends in explainable AI and quantum computing applications"
            }

        # Peer review response
        elif "peer review" in prompt_lower or "critique" in prompt_lower or "rate this question" in prompt_lower:
            mock_response = {
                "overall_score": 7,
                "strengths": ["Novel synthesis of concepts", "Clear practical applications", "Good specificity"],
                "weaknesses": ["Data requirements might be high", "Requires specialized expertise", "Validation challenging"],
                "suggested_improvements": ["Consider more accessible data sources", "Propose pilot study first", "Include computational feasibility analysis"],
                "approve": True
            }

        # Data analysis response
        elif "data" in prompt_lower or "insight" in prompt_lower or "analyze" in prompt_lower:
            mock_response = [
                "Data shows strong correlation between model complexity and interpretability trade-offs",
                "Quantum approaches show advantage in specific molecular prediction tasks",
                "Hybrid methods consistently outperform pure quantum or classical approaches",
                "Data availability is a major bottleneck for validation studies",
                "Open source implementations are critical for reproducibility"
            ]

        # Default response for other queries
        else:
            mock_response = {
                "response": f"This is a mock response for task type: {task_type}",
                "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "success": True
            }

        return {
            "content": json.dumps(mock_response) if not isinstance(mock_response, str) else mock_response,
            "confidence_score": kwargs.get("temperature", 0.7) * 0.8 + 0.2,  # Simulate confidence based on temperature
            "success": True,
            "provider": "mock",
            "model": "mock-llm"
        }


class GroqProvider(BaseProvider):
    async def generate(self, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """Generate using Groq API with Llama 3.1 70B (free tier)"""
        import aiohttp

        if not hasattr(Config, 'GROQ_API_KEY') or not Config.GROQ_API_KEY:
            raise ValueError("Groq API key not configured")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Map task_type to appropriate model and parameters
        model_map = {
            "creative": "llama-3.3-70b-versatile",  # Updated to current model
            "analysis": "llama-3.3-70b-versatile",
            "writing": "llama-3.3-70b-versatile",
            "general": "llama-3.3-70b-versatile"
        }

        model = model_map.get(task_type, "llama-3.1-70b-versatile")

        # System prompt based on task type
        system_prompts = {
            "creative": "You are a creative AI research assistant. Generate novel, innovative ideas and questions.",
            "analysis": "You are an analytical AI research assistant. Provide thorough, logical analysis.",
            "writing": "You are a scientific writer. Write clear, concise, professional research content.",
            "general": "You are a helpful AI research assistant."
        }

        system_prompt = system_prompts.get(task_type, "You are a helpful AI assistant.")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "top_p": 1,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data["choices"][0]["message"]["content"]

                        # Try to parse as JSON if it looks like JSON
                        try:
                            import json
                            # Check if content is JSON
                            if content.strip().startswith('{') or content.strip().startswith('['):
                                parsed = json.loads(content)
                                return {
                                    "content": json.dumps(parsed),
                                    "confidence_score": 0.9,
                                    "success": True,
                                    "provider": "groq",
                                    "model": model
                                }
                        except:
                            pass

                        # Return as text
                        return {
                            "content": content,
                            "confidence_score": 0.9,
                            "success": True,
                            "provider": "groq",
                            "model": model
                        }
                    else:
                        error_text = await response.text()
                        print(f"Groq API error: {response.status} - {error_text}")
                        raise Exception(f"Groq API error: {response.status}")
        except Exception as e:
            print(f"Groq API call failed: {e}")
            raise  # Re-raise to try next provider


# Optional: Add a simple OpenAI provider (if you have free credits)
class OpenAIProvider(BaseProvider):
    async def generate(self, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """Generate using OpenAI API (if you have free credits)"""
        # Similar implementation to GroqProvider but for OpenAI
        # For now, just raise to use mock
        raise Exception("OpenAI provider not implemented")


# Optional: Add Anthropic provider
class AnthropicProvider(BaseProvider):
    async def generate(self, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """Generate using Anthropic Claude API"""
        raise Exception("Anthropic provider not implemented")


# Global instance
llm_client = LLMClient()