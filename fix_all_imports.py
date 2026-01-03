# fix_all_imports.py
import os


def fix_all_agent_files():
    """Fix ALL agent files to avoid circular imports"""

    # List all agent files
    agent_files = [
        'agents/domain_scout.py',
        'agents/question_generator.py',
        'agents/data_alchemist.py',
        'agents/experiment_designer.py',
        'agents/critic.py',
        'agents/uncertainty_agent.py',
        'agents/orchestrator.py',
        'agents/base_agent.py',
        'core/llm_client.py',
        'core/experiment_runner.py',
        'workflows/research_workflow.py'
    ]

    # Template for fixed imports
    import_template = '''"""
{module_doc}
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging  # ✅ Use standard logging initially

# Create module-level logger that will be setup later
logger = logging.getLogger(__name__)

{rest_of_imports}'''

    for file_path in agent_files:
        if os.path.exists(file_path):
            print(f"🔧 Fixing {file_path}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip if already has standard logging
            if 'import logging' in content and 'logger = logging.getLogger(__name__)' in content:
                print(f"  ✓ Already fixed")
                continue

            # Extract module docstring if exists
            module_doc = ""
            if content.startswith('"""'):
                end_doc = content.find('"""', 3)
                if end_doc != -1:
                    module_doc = content[3:end_doc].strip()

            # Find the rest after imports
            lines = content.split('\n')

            # Remove problematic logger setup lines
            new_lines = []
            for line in lines:
                # Remove these problematic lines
                if 'logger = setup_logger(__name__)' in line:
                    continue
                if 'from utils.logger import setup_logger' in line and 'import logging' not in content:
                    continue
                new_lines.append(line)

            # Reconstruct with proper imports at top
            fixed_content = import_template.format(
                module_doc=module_doc if module_doc else "Module",
                rest_of_imports='\n'.join(new_lines)
            )

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print(f"  ✓ Fixed")
        else:
            print(f"  ⚠️ File not found: {file_path}")


if __name__ == '__main__':
    fix_all_agent_files()
    print("\n✅ ALL files fixed! Now run the test again.")