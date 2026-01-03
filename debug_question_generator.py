import asyncio
import sys
import os
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.question_generator import QuestionGeneratorAgent


async def debug_question_generator():
    agent = QuestionGeneratorAgent()

    # Test context
    context = {
        "domain": "Quantum Machine Learning for Drug Discovery",
        "domain_data": {
            "selected": {
                "name": "Quantum Machine Learning for Drug Discovery",
                "description": "Applying quantum computing principles to machine learning for pharmaceutical research",
                "evidence": ["Recent arXiv papers", "Increased GitHub activity", "2024 conference focus"]
            }
        }
    }

    print("Testing question generation...")

    # First, let's test the LLM directly
    domain = context["domain"]
    prompt = f"""Based on this research domain: "{domain}"

Generate 3 novel research questions that require synthesis of different concepts.

Return a JSON array of questions. Each question should have:
- question: the research question text
- novelty_score: 1-10 rating of novelty
- feasibility: low, medium, or high

Example:
[
  {{
    "question": "How can concept A be combined with concept B to solve problem C?",
    "novelty_score": 8,
    "feasibility": "medium"
  }}
]

Now generate 3 questions about {domain}:"""

    print(f"\nSending prompt to LLM...")
    print(f"Prompt preview: {prompt[:200]}...")

    try:
        response = await agent.llm.generate(prompt, task_type="creative")
        content = response.get("content", "")
        print(f"\nRaw LLM response ({len(content)} chars):")
        print("-" * 40)
        print(content[:1000])  # Show first 1000 chars
        print("-" * 40)

        # Try to parse
        print("\nTrying to parse JSON...")

        # Method 1: Direct parse
        try:
            parsed = json.loads(content)
            print(f"✓ Successfully parsed as JSON")
            print(f"  Type: {type(parsed)}")
            if isinstance(parsed, dict):
                print(f"  Keys: {list(parsed.keys())}")
                if "questions" in parsed:
                    questions = parsed["questions"]
                    print(f"  Found {len(questions)} questions in 'questions' key")
                else:
                    print(f"  No 'questions' key found")
            elif isinstance(parsed, list):
                print(f"  It's a list with {len(parsed)} items")
                questions = parsed
            else:
                print(f"  Unexpected type: {type(parsed)}")
                questions = []
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            questions = []

        # Method 2: Extract JSON from text
        if not questions:
            print("\nTrying to extract JSON from text...")
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                print(f"✓ Found potential JSON array in text")
                try:
                    parsed = json.loads(json_match.group())
                    if isinstance(parsed, list):
                        questions = parsed
                        print(f"✓ Successfully parsed extracted JSON with {len(questions)} questions")
                    else:
                        print(f"✗ Extracted text is not a list")
                except Exception as e:
                    print(f"✗ Failed to parse extracted text: {e}")
            else:
                print(f"✗ No JSON array found in text")

        print(f"\nTotal questions found after parsing: {len(questions)}")

        if questions:
            print("\nFirst question:")
            print(f"  Question: {questions[0].get('question', 'No question')[:100]}...")
            print(f"  Novelty: {questions[0].get('novelty_score', 'N/A')}")
            print(f"  Feasibility: {questions[0].get('feasibility', 'N/A')}")

        # Now test the agent's method
        print("\n" + "=" * 60)
        print("Testing agent._generate_questions() method...")
        questions = await agent._generate_questions(context["domain"], context["domain_data"]["selected"])
        print(f"Agent returned {len(questions)} questions")

        for i, q in enumerate(questions[:2]):
            print(f"\nQuestion {i + 1}:")
            print(f"  Text: {q.get('question', 'No question')[:80]}...")
            print(f"  Novelty: {q.get('novelty_score', 'N/A')}")
            print(f"  Feasibility: {q.get('feasibility', 'N/A')}")

        # Test full execution
        print("\n" + "=" * 60)
        print("Testing full execute() method...")
        result = await agent.execute(context)
        print(f"Success: {result.get('success', False)}")
        print(f"Message: {result.get('message', 'No message')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")

        if result.get("success", False):
            data = result.get("data", {})
            questions = data.get("questions", [])
            print(f"Generated {len(questions)} questions")
            if questions:
                selected = data.get("selected_question", {})
                print(f"Selected question: {selected.get('question', 'None')[:100]}...")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_question_generator())