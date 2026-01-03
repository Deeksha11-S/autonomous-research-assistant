"""
Test configuration for agents
Create .env file with test API keys
"""

import os
from dotenv import load_dotenv

load_dotenv()


class TestConfig:
    # LLM API Keys (use test keys or free tiers)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "test_key")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "test_key")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "test_key")
    HF_TOKEN = os.getenv("HF_TOKEN", "test_token")

    # Search API Keys
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "test_key")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "test_key")

    # Test settings
    TEST_MODE = True
    USE_MOCK_DATA = True  # Use mock data for testing
    MAX_TEST_ITERATIONS = 1

    # Memory settings
    CHROMA_PERSIST_DIR = "./test_chroma_db"

    # Agent settings
    MIN_CONFIDENCE_THRESHOLD = 0.5  # Lower for testing
    MAX_ITERATIONS = 2  # Fewer for testing

    @classmethod
    def validate(cls):
        """Validate configuration"""
        missing = []

        if cls.GROQ_API_KEY == "test_key":
            print("⚠️ Using test GROQ API key")
            missing.append("GROQ_API_KEY")

        if cls.TAVILY_API_KEY == "test_key":
            print("⚠️ Using test TAVILY API key")
            missing.append("TAVILY_API_KEY")

        if missing:
            print(f"\n🔑 Missing API keys: {', '.join(missing)}")
            print("Some tests may fail or use mock data.")
            print("Create a .env file with your API keys.")

        return len(missing) == 0


# Update utils/config.py to use test config in test mode
import sys

if "pytest" in sys.modules or "test_agents" in sys.argv[0]:
    from utils import config

    config.Config = TestConfig