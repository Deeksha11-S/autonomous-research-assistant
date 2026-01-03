import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import llm_client

async def test():
    response = await llm_client.generate("Test prompt", task_type="analysis")
    print(f"Type of response: {type(response)}")
    print(f"Response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
    print(f"Has 'content' key: {'content' in response}")
    print(f"Content type: {type(response.get('content'))}")
    print(f"First 200 chars of content: {str(response.get('content', ''))[:200]}")

if __name__ == "__main__":
    asyncio.run(test())