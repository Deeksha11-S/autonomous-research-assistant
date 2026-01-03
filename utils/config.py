import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM Settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Experiment Settings
    MIN_CONFIDENCE_THRESHOLD = 0.6  # 60% confidence required
    MAX_ITERATIONS = 5
    MIN_SAMPLE_SIZE = 30

    # Memory Settings
    CHROMA_PERSIST_DIR = "./data/chroma_db"

    # Search Settings
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")