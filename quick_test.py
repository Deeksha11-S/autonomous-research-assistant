#!/usr/bin/env python3
"""
Quick test to verify fixes
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


async def test_fixes():
    print("🧪 Testing fixes...")

    # Test 1: Import agents
    try:
        from agents.base_agent import BaseAgent
        from agents.question_generator import QuestionGeneratorAgent
        from agents.data_alchemist import DataAlchemistAgent
        print("✅ All agent imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Create agents
    try:
        qg_agent = QuestionGeneratorAgent()
        da_agent = DataAlchemistAgent()
        print(f"✅ Agent creation: {qg_agent.name}, {da_agent.name}")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

    # Test 3: Test LLM generation
    try:
        response = await qg_agent.llm_generate("Test prompt")
        if isinstance(response, dict) and "content" in response:
            print("✅ LLM generation working")
        else:
            print("❌ LLM response format incorrect")
            return False
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        return False

    # Test 4: Test question generation
    try:
        context = {
            "domain": "Test Domain",
            "domain_data": {
                "selected": {
                    "name": "Quantum Machine Learning",
                    "description": "Test description",
                    "evidence": ["test1", "test2"]
                }
            }
        }
        result = await qg_agent.execute(context)
        print(f"✅ Question generation executed: {result.get('success', False)}")
    except Exception as e:
        print(f"❌ Question generation execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ All basic tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    sys.exit(0 if success else 1)