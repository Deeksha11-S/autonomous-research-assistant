# run_research.py
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.domain_scout import DomainScoutAgent
from agents.question_generator import QuestionGeneratorAgent
from agents.data_alchemist import DataAlchemistAgent
from agents.experiment_designer import ExperimentDesignerAgent
from agents.critic import CriticAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.orchestrator import OrchestratorAgent


async def run_complete_research():
    print("🚀 Starting Complete Research Workflow")
    print("=" * 60)

    # Initialize agents
    domain_scout = DomainScoutAgent()
    question_gen = QuestionGeneratorAgent()
    data_alchemist = DataAlchemistAgent()
    experiment_designer = ExperimentDesignerAgent()
    critic = CriticAgent()
    uncertainty = UncertaintyAgent()
    orchestrator = OrchestratorAgent()

    context = {}

    # Step 1: Domain scouting
    print("\n1. 🔍 Domain Scouting...")
    result = await domain_scout.execute(context)
    if result.get("success"):
        domains = result["data"].get("domains", [])
        if domains:
            selected_domain = domains[0]
            context["domain"] = selected_domain.get("name")
            context["domain_data"] = {"selected": selected_domain}
            print(f"   Selected: {context['domain']}")

    # Step 2: Question generation
    print("\n2. ❓ Question Generation...")
    result = await question_gen.execute(context)
    if result.get("success"):
        questions = result["data"].get("questions", [])
        if questions:
            selected_question = result["data"].get("selected_question", questions[0])
            context["selected_question"] = selected_question.get("question")
            context["question_data"] = {"selected": selected_question}
            print(f"   Selected: {context['selected_question'][:80]}...")

    # Step 3: Data gathering
    print("\n3. ⚗️ Data Gathering...")
    result = await data_alchemist.execute(context)
    if result.get("success"):
        context["data_sources"] = result["data"].get("data_sources", [])
        context["insights"] = result["data"].get("insights", [])
        print(f"   Gathered {len(context['data_sources'])} data sources")
        print(f"   Extracted {len(context['insights'])} insights")

    # Step 4: Experiment design
    print("\n4. 🧪 Experiment Design...")
    result = await experiment_designer.execute(context)
    if result.get("success"):
        context["hypothesis"] = result["data"].get("hypothesis", {})
        context["experiments"] = result["data"].get("experiments", [])
        print(f"   Hypothesis: {context['hypothesis'].get('alternative_hypothesis', '')[:80]}...")
        print(f"   Designed {len(context['experiments'])} experiments")

    # Step 5: Critique
    print("\n5. 👓 Research Critique...")
    context["iteration"] = 1
    result = await critic.execute(context)
    if result.get("success"):
        context["critique_results"] = result["data"]
        needs_iteration = result["data"].get("iteration_decision", {}).get("needs_iteration", False)
        print(f"   Needs iteration: {'YES' if needs_iteration else 'NO'}")

    # Step 6: Uncertainty assessment
    print("\n6. 📊 Uncertainty Assessment...")
    result = await uncertainty.execute(context)
    if result.get("success"):
        overall_confidence = result["data"].get("overall_confidence", 0)
        should_abstain = result["data"].get("should_abstain", True)
        print(f"   Overall confidence: {overall_confidence:.1%}")
        print(f"   Should proceed: {'YES' if not should_abstain else 'NO'}")

    # Step 7: Orchestration
    print("\n7. 🚀 Research Orchestration...")
    result = await orchestrator.execute(context)
    if result.get("success"):
        print("   Research orchestration complete!")

        # Generate paper
        paper = await orchestrator._generate_research_paper()
        print(f"   Paper generated: {len(paper)} characters")

        # Save paper
        with open("research_paper.md", "w", encoding="utf-8") as f:
            f.write(paper)
        print("   Paper saved to: research_paper.md")

    print("\n" + "=" * 60)
    print("✅ Research workflow completed!")


if __name__ == "__main__":
    asyncio.run(run_complete_research())