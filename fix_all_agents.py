#!/usr/bin/env python3
"""
Fix all agent files to handle dictionary responses correctly
"""
import os
import re


def fix_experiment_designer():
    """Fix experiment_designer.py"""
    file_path = "agents/experiment_designer.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix the _formulate_hypothesis method
    old_code = '''        try:
            response = await self.llm.generate(formulation_prompt, task_type="analysis")
            hypothesis = json.loads(response.content)
            hypothesis["formulation_confidence"] = response.confidence_score
            return hypothesis'''

    new_code = '''        try:
            response = await self.llm.generate(formulation_prompt, task_type="analysis")
            # The response is a dictionary, not an object
            content = response.get("content", "{}")
            confidence = response.get("confidence_score", 0.7)

            try:
                hypothesis = json.loads(content)
            except json.JSONDecodeError:
                # If content is already a dict or not valid JSON
                if isinstance(content, dict):
                    hypothesis = content
                else:
                    # Try to extract JSON from string
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        hypothesis = json.loads(json_match.group())
                    else:
                        hypothesis = self._create_default_hypothesis(question)

            hypothesis["formulation_confidence"] = confidence
            return hypothesis'''

    content = content.replace(old_code, new_code)

    # Fix other occurrences
    content = re.sub(r'json\.loads\(response\.content\)',
                     '''content = response.get("content", "{}")
             try:
                 json.loads(content)''', content)

    content = re.sub(r'response\.content', 'response.get("content", "{}")', content)
    content = re.sub(r'response\.confidence_score', 'response.get("confidence_score", 0.7)', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Fixed experiment_designer.py")


def fix_critic():
    """Fix critic.py"""
    file_path = "agents/critic.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all problematic patterns
    content = re.sub(r'json\.loads\(response\.content\)',
                     '''content = response.get("content", "{}")
             try:
                 json.loads(content)''', content)

    content = re.sub(r'response\.content', 'response.get("content", "{}")', content)
    content = re.sub(r'response\.confidence_score', 'response.get("confidence_score", 0.7)', content)

    # Fix specific method patterns
    content = re.sub(
        r'response = await self\.llm\.generate\(critique_prompt, task_type="analysis", max_tokens=2500\)\s*\n\s*critique = json\.loads\(response\.content\)',
        '''response = await self.llm.generate(critique_prompt, task_type="analysis", max_tokens=2500)
            content = response.get("content", '{"issues": [], "severity": "low"}')
            try:
                critique = json.loads(content)
            except json.JSONDecodeError:
                critique = {"issues": ["JSON parse error"], "severity": "medium"}''',
        content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Fixed critic.py")


def fix_uncertainty_agent():
    """Fix uncertainty_agent.py"""
    file_path = "agents/uncertainty_agent.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all problematic patterns
    content = re.sub(r'json\.loads\(response\.content\)',
                     '''content = response.get("content", "{}")
             try:
                 json.loads(content)''', content)

    content = re.sub(r'response\.content', 'response.get("content", "{}")', content)
    content = re.sub(r'response\.confidence_score', 'response.get("confidence_score", 0.7)', content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Fixed uncertainty_agent.py")


def fix_base_agent():
    """Fix base_agent.py to ensure consistent response handling"""
    file_path = "agents/base_agent.py"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if we need to add helper method
    if 'def safe_llm_response' not in content:
        # Add a helper method at the end of the class
        helper_method = '''
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
'''

        # Insert before the last line (closing of class)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == 'async def store_in_memory(self, content: str, metadata: Dict[str, Any] = None):':
                insert_index = i
                break

        # Insert helper method
        lines.insert(insert_index, helper_method)
        content = '\n'.join(lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Updated base_agent.py")


def create_safe_llm_wrapper():
    """Create a safe wrapper for LLM responses"""
    wrapper_code = '''"""
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
            json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
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
'''

    with open("utils/safe_llm.py", 'w', encoding='utf-8') as f:
        f.write(wrapper_code)

    print("✅ Created safe_llm.py utility")


def main():
    """Fix all agent files"""
    print("🔧 Fixing agent files...")

    # Create utils directory if it doesn't exist
    if not os.path.exists("utils"):
        os.makedirs("utils")

    create_safe_llm_wrapper()
    fix_base_agent()
    fix_experiment_designer()
    fix_critic()
    fix_uncertainty_agent()

    print("\n✅ All fixes applied!")
    print("\nNow run: python test_agents.py --agent experiment")


if __name__ == "__main__":
    main()