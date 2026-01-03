"""
Test script for individual agent verification
Run: python test_agents.py
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.domain_scout import DomainScoutAgent
from agents.question_generator import QuestionGeneratorAgent
from agents.data_alchemist import DataAlchemistAgent
from agents.experiment_designer import ExperimentDesignerAgent
from agents.critic import CriticAgent
from agents.uncertainty_agent import UncertaintyAgent
from agents.orchestrator import OrchestratorAgent

# Mock context data for testing
MOCK_CONTEXT = {
    "domain": "Quantum Machine Learning for Drug Discovery",
    "selected_question": "How can quantum-inspired algorithms improve drug repurposing efficiency for rare diseases?",
    "data_sources": [
        {
            "type": "research_paper",
            "source": "arXiv",
            "content": "Quantum machine learning shows 30% improvement in drug binding prediction accuracy compared to classical methods.",
            "metadata": {"year": 2024, "citations": 45}
        },
        {
            "type": "dataset",
            "source": "GitHub",
            "content": "Dataset of 10,000 drug-protein interactions with quantum chemical descriptors",
            "metadata": {"size": "500MB", "records": 10000}
        }
    ],
    "insights": [
        "Quantum algorithms can process molecular data 10x faster than classical methods",
        "Current limitations include noise in quantum hardware and small qubit counts",
        "Hybrid quantum-classical approaches show most promise for near-term applications"
    ],
    "hypothesis": {
        "null_hypothesis": "Quantum-inspired algorithms provide no improvement in drug repurposing efficiency",
        "alternative_hypothesis": "Quantum-inspired algorithms significantly improve drug repurposing efficiency by 20% or more",
        "variables": [
            {"name": "algorithm_type", "type": "independent", "values": ["quantum", "classical"]},
            {"name": "prediction_accuracy", "type": "dependent", "measurement": "percentage"}
        ]
    },
    "iteration": 1,
    "confidence_scores": {}
}


class AgentTester:
    def __init__(self):
        self.results = {}
        self.test_count = 0
        self.passed_count = 0

    async def test_domain_scout(self):
        """Test DomainScoutAgent"""
        print("\n" + "=" * 60)
        print("🔍 TESTING DOMAIN SCOUT AGENT")
        print("=" * 60)

        try:
            agent = DomainScoutAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")

            # Test with empty context
            print("\n📋 Test 1: Empty context")
            result = await agent.execute({})
            self._print_result("Empty context", result)

            # Test with some context
            print("\n📋 Test 2: With research context")
            context = {"research_area": "emerging technologies"}
            result = await agent.execute(context)
            self._print_result("With context", result)

            # Validate response structure
            required_keys = ["success", "confidence", "should_abstain", "data"]
            if all(key in result for key in required_keys):
                print("✅ Response structure: VALID")
                if result["success"]:
                    domains = result["data"].get("domains", [])
                    print(f"✅ Discovered {len(domains)} domains")
                    for i, domain in enumerate(domains[:3], 1):
                        print(f"  {i}. {domain.get('name', 'Unknown')}")
                else:
                    print("❌ Agent failed or abstained")
            else:
                print("❌ Response structure: INVALID")

            self.results["domain_scout"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_question_generator(self):
        """Test QuestionGeneratorAgent"""
        print("\n" + "=" * 60)
        print("❓ TESTING QUESTION GENERATOR AGENT")
        print("=" * 60)

        try:
            agent = QuestionGeneratorAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")

            # Prepare context
            context = {
                "domain": "Quantum Machine Learning",
                "domain_data": {
                    "selected": {
                        "name": "Quantum Machine Learning for Drug Discovery",
                        "description": "Applying quantum computing principles to machine learning for pharmaceutical research",
                        "evidence": ["Recent arXiv papers", "Increased GitHub activity", "2024 conference focus"]
                    }
                }
            }

            print("\n📋 Test: Generate research questions")
            result = await agent.execute(context)
            self._print_result("Question generation", result)

            # Validate questions
            if result.get("success", False):
                questions = result["data"].get("questions", [])
                print(f"✅ Generated {len(questions)} questions")

                for i, q in enumerate(questions[:3], 1):
                    print(f"\n  Question {i}:")
                    print(f"    Text: {q.get('question', 'No question')[:80]}...")
                    print(f"    Novelty: {q.get('novelty_score', 'N/A')}/10")
                    print(f"    Feasibility: {q.get('feasibility', 'N/A')}")

                # Test question selection
                print("\n📋 Test: Select best question")
                selection = await agent.select_best_question(questions)
                if selection.get("selected_question"):
                    print(f"✅ Selected: {selection['selected_question'].get('question', 'Unknown')[:100]}...")
                else:
                    print("❌ No question selected")
            else:
                print("❌ Question generation failed")

            self.results["question_generator"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_data_alchemist(self):
        """Test DataAlchemistAgent"""
        print("\n" + "=" * 60)
        print("⚗️ TESTING DATA ALCHEMIST AGENT")
        print("=" * 60)

        try:
            agent = DataAlchemistAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")

            # Prepare context
            context = {
                "selected_question": "How can quantum algorithms improve drug discovery?",
                "domain": "Quantum Computing in Pharmacology"
            }

            print("\n📋 Test: Data gathering and processing")
            result = await agent.execute(context)
            self._print_result("Data alchemy", result)

            # Validate data sources
            if result.get("success", False):
                data = result["data"]
                sources = data.get("data_sources", [])
                insights = data.get("insights", [])

                print(f"✅ Gathered {len(sources)} data sources")
                print(f"✅ Extracted {len(insights)} insights")

                # Show source types
                source_types = {}
                for source in sources[:5]:
                    stype = source.get("type", "unknown")
                    source_types[stype] = source_types.get(stype, 0) + 1

                print("\n📊 Source types:")
                for stype, count in source_types.items():
                    print(f"  {stype}: {count}")

                # Show insights
                print("\n💡 Key insights:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"  {i}. {insight[:100]}...")
            else:
                print(f"❌ Data gathering failed: {result.get('message', 'Unknown error')}")

            self.results["data_alchemist"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_experiment_designer(self):
        """Test ExperimentDesignerAgent"""
        print("\n" + "=" * 60)
        print("🧪 TESTING EXPERIMENT DESIGNER AGENT")
        print("=" * 60)

        try:
            agent = ExperimentDesignerAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")

            # Prepare context
            context = {
                "selected_question": "Impact of quantum algorithms on drug discovery accuracy",
                "data_insights": [
                    "Quantum methods show 30% improvement in certain cases",
                    "Current limitations include hardware constraints",
                    "Hybrid approaches are most practical"
                ],
                "data_sources": [
                    {"type": "research_paper", "source": "Nature Quantum", "content": "Study on quantum advantage"},
                    {"type": "dataset", "source": "MoleculeNet", "content": "Drug interaction data"}
                ]
            }

            print("\n📋 Test: Experiment design")
            result = await agent.execute(context)
            self._print_result("Experiment design", result)

            # Validate experiments
            if result.get("success", False):
                data = result["data"]
                hypothesis = data.get("hypothesis", {})
                experiments = data.get("experiments", [])

                print(f"✅ Hypothesis formulated:")
                print(f"   H₀: {hypothesis.get('null_hypothesis', 'None')[:80]}...")
                print(f"   H₁: {hypothesis.get('alternative_hypothesis', 'None')[:80]}...")

                print(f"\n✅ Designed {len(experiments)} experiments:")
                for i, exp in enumerate(experiments[:2], 1):
                    print(f"\n  Experiment {i}: {exp.get('name', 'Unnamed')}")
                    print(f"    Type: {exp.get('type', 'Unknown')}")
                    print(f"    Sample size: {exp.get('sample_size', 'N/A')}")
                    print(f"    Complexity: {exp.get('complexity', 'Unknown')}")

                    # Show validation
                    validation = exp.get("validation", {})
                    if validation.get("is_valid", False):
                        print(f"    ✅ VALID (score: {validation.get('score', 0):.2f})")
                    else:
                        print(f"    ❌ INVALID")
                        for issue in validation.get("critical_issues", [])[:2]:
                            print(f"      - {issue[:60]}...")
            else:
                print("❌ Experiment design failed")

            self.results["experiment_designer"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_critic(self):
        """Test CriticAgent"""
        print("\n" + "=" * 60)
        print("👓 TESTING CRITIC AGENT")
        print("=" * 60)

        try:
            agent = CriticAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")

            # Prepare mock experiment data
            context = {
                "iteration": 1,
                "hypothesis": {
                    "null_hypothesis": "No effect of treatment",
                    "alternative_hypothesis": "Treatment improves outcomes by 20%",
                    "variables": [
                        {"name": "treatment", "type": "independent"},
                        {"name": "outcome", "type": "dependent"}
                    ]
                },
                "experiments": [
                    {
                        "name": "A/B Test",
                        "type": "ab_test",
                        "sample_size": 50,
                        "procedure": "Random assignment to control and treatment groups",
                        "design": "Basic A/B test without blinding",
                        "metrics": ["success_rate", "p_value"]
                    }
                ],
                "experiment_results": [
                    {
                        "experiment_name": "A/B Test",
                        "p_value": 0.06,
                        "effect_size": 0.15,
                        "confidence_interval": [0.05, 0.25],
                        "claimed_significant": True
                    }
                ],
                "data_sources": [
                    {"type": "synthetic", "source": "test", "content": "test data"}
                ]
            }

            print("\n📋 Test: Research critique")
            result = await agent.execute(context)
            self._print_result("Critique", result)

            # Validate critique
            if result.get("success", False):
                data = result["data"]
                needs_iteration = data.get("iteration_decision", {}).get("needs_iteration", False)

                print(f"✅ Critique complete")
                print(f"   Needs iteration: {'YES' if needs_iteration else 'NO'}")

                # Show methodology issues
                method_issues = data.get("methodology_critique", {}).get("issues", [])
                if method_issues:
                    print(f"\n📋 Methodology issues ({len(method_issues)}):")
                    for i, issue in enumerate(method_issues[:2], 1):
                        if isinstance(issue, dict):
                            print(f"  {i}. {issue.get('description', 'Issue')[:80]}...")
                        else:
                            print(f"  {i}. {str(issue)[:80]}...")

                # Show statistical issues
                stat_checks = data.get("statistical_critique", {}).get("automated_checks", [])
                if stat_checks:
                    print(f"\n📊 Statistical issues ({len(stat_checks)}):")
                    for check in stat_checks[:2]:
                        if isinstance(check, dict):
                            print(f"  - {check.get('issue', 'Issue')[:80]}...")
                        else:
                            print(f"  - {str(check)[:80]}...")

                # Show feedback preview
                feedback = data.get("feedback", "")
                if feedback:
                    print(f"\n💬 Feedback preview:")
                    lines = feedback.split('\n')[:5]
                    for line in lines:
                        print(f"  {line}")
            else:
                print("❌ Critique failed")

            self.results["critic"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_uncertainty_agent(self):
        """Test UncertaintyAgent"""
        print("\n" + "=" * 60)
        print("📊 TESTING UNCERTAINTY AGENT")
        print("=" * 60)

        try:
            agent = UncertaintyAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")
            print(f"Confidence threshold: {agent.confidence_threshold:.0%}")

            # Prepare context with component data
            context = {
                "iteration": 1,
                "domain_data": {
                    "selected": {
                        "name": "Test Domain",
                        "confidence": 0.7
                    }
                },
                "question_data": {
                    "selected": {
                        "question": "Test question",
                        "novelty_score": 8
                    }
                },
                "data_sources": [
                    {"type": "research_paper", "source": "test", "content": "test content"},
                    {"type": "dataset", "source": "test", "content": "test data"}
                ],
                "hypothesis": {
                    "testability_score": 7,
                    "variables": [{"name": "test"}]
                },
                "experiments": [
                    {
                        "name": "Test Experiment",
                        "validation": {"is_valid": True, "score": 0.8},
                        "sample_size": 100
                    }
                ],
                "critique_results": {
                    "needs_iteration": False,
                    "total_issues": 2,
                    "severity_level": 1
                },
                "experiment_results": [
                    {
                        "p_value": 0.03,
                        "effect_size": 0.4,
                        "confidence_interval": [0.3, 0.5]
                    }
                ]
            }

            print("\n📋 Test: Uncertainty quantification")
            result = await agent.execute(context)
            self._print_result("Uncertainty assessment", result)

            # Validate assessment
            if result.get("success", False):
                data = result["data"]
                overall_confidence = data.get("overall_confidence", 0)
                should_abstain = data.get("should_abstain", True)

                print(f"✅ Overall confidence: {overall_confidence:.1%}")
                print(f"✅ Should abstain: {'YES' if should_abstain else 'NO'}")
                print(
                    f"✅ Meets threshold ({agent.confidence_threshold:.0%}): {'YES' if overall_confidence >= agent.confidence_threshold else 'NO'}")

                # Show component breakdown
                components = data.get("component_uncertainties", {})
                if components:
                    print(f"\n📈 Component confidence:")
                    for comp, assessment in components.items():
                        confidence = assessment.get("calibrated_confidence", assessment.get("confidence", 0))
                        status = "✅" if confidence >= 0.6 else "⚠️"
                        print(f"  {status} {comp}: {confidence:.1%}")

                # Show critical uncertainties
                critical = data.get("critical_uncertainty_sources", [])
                if critical:
                    print(f"\n⚠️ Critical uncertainties ({len(critical)}):")
                    for unc in critical:
                        print(f"  - {unc['component']}: {unc['confidence']:.1%} ({unc['impact']})")

                # Show metrics
                metrics = data.get("confidence_metrics", {})
                if metrics:
                    print(f"\n📋 Assessment metrics:")
                    print(f"  Reliability: {metrics.get('reliability', 'unknown')}")
                    print(f"  CI width: {metrics.get('interval_width', 0):.3f}")

                # Show report preview
                report = data.get("uncertainty_report", "")
                if report:
                    lines = report.split('\n')[:8]
                    print(f"\n📄 Report preview:")
                    for line in lines:
                        print(f"  {line}")
            else:
                print("❌ Uncertainty assessment failed")

            self.results["uncertainty_agent"] = result
            return result

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_orchestrator(self):
        """Test OrchestratorAgent"""
        print("\n" + "=" * 60)
        print("🚀 TESTING ORCHESTRATOR AGENT")
        print("=" * 60)

        try:
            agent = OrchestratorAgent()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.description}")
            print(f"Max iterations: {agent.max_iterations}")

            # Test quick run (1 iteration)
            print("\n📋 Test: Quick research run (1 iteration)")

            # Mock some results to avoid full execution
            agent.research_state = {
                "domain": "Test Domain",
                "selected_question": "Test Question",
                "data_sources": [{"type": "test", "source": "test"}],
                "confidence_scores": {"iteration_1": 0.75},
                "agent_messages": ["Test message"]
            }

            # Test status check
            print("\n📋 Test: Get research status")
            status = agent.get_research_status()
            print(f"✅ Status retrieved")
            print(f"   Current iteration: {status['current_iteration']}")
            print(f"   Domain: {status['research_state'].get('domain', 'None')}")
            print(f"   Confidence scores: {status['confidence_scores']}")

            # Test conflict resolution
            print("\n📋 Test: Conflict resolution")
            conflict = {
                "agent1_position": "Use method A for analysis",
                "agent2_position": "Use method B for analysis",
                "context": "Statistical analysis method selection"
            }

            resolution = await agent.resolve_conflict("DataAlchemist", "ExperimentDesigner", conflict)
            if resolution.get("success", False):
                print(f"✅ Conflict resolved successfully")
                print(f"   Resolution ID: {resolution.get('conflict_id', 'N/A')}")
            else:
                print(f"⚠️ Conflict resolution failed: {resolution.get('error', 'Unknown')}")

            # Test paper generation
            print("\n📋 Test: Research paper generation")
            try:
                paper = await agent._generate_research_paper()
                if paper and len(paper) > 100:
                    print(f"✅ Paper generated ({len(paper)} chars)")

                    # Show paper structure
                    lines = paper.split('\n')[:10]
                    print(f"\n📄 Paper preview (first 10 lines):")
                    for i, line in enumerate(lines, 1):
                        print(f"  {i:2}. {line}")
                else:
                    print("❌ Paper generation failed or too short")
            except Exception as e:
                print(f"❌ Paper generation test failed: {e}")

            self.results["orchestrator"] = status
            return status

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def test_integration(self):
        """Test basic integration between agents"""
        print("\n" + "=" * 60)
        print("🔗 TESTING BASIC INTEGRATION")
        print("=" * 60)

        try:
            # Create agents
            domain_scout = DomainScoutAgent()
            question_gen = QuestionGeneratorAgent()

            # Test data flow
            print("\n📋 Test: Domain → Question flow")

            # Get domains
            domain_result = await domain_scout.execute({})
            if domain_result.get("success", False):
                domains = domain_result["data"].get("domains", [])
                if domains:
                    test_domain = domains[0]

                    # Use domain to generate questions
                    context = {
                        "domain": test_domain.get("name", "Test Domain"),
                        "domain_data": {
                            "selected": test_domain
                        }
                    }

                    question_result = await question_gen.execute(context)
                    if question_result.get("success", False):
                        questions = question_result["data"].get("questions", [])
                        print(f"✅ Integration successful!")
                        print(f"   Domain: {test_domain.get('name', 'Unknown')}")
                        print(f"   Questions generated: {len(questions)}")

                        if questions:
                            # Test question selection
                            selection = await question_gen.select_best_question(questions)
                            if selection.get("selected_question"):
                                selected = selection["selected_question"]
                                print(f"   Selected question: {selected.get('question', 'Unknown')[:80]}...")
                            else:
                                print("   ❌ Question selection failed")
                    else:
                        print("❌ Question generation failed in integration test")
                else:
                    print("❌ No domains found for integration test")
            else:
                print("❌ Domain scouting failed in integration test")

            return True

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False

    def _print_result(self, test_name: str, result: Dict):
        """Print test result in formatted way"""
        print(f"\n📊 {test_name.upper()} RESULT:")
        print(f"   Success: {'✅' if result.get('success', False) else '❌'}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Should abstain: {'✅ YES' if result.get('should_abstain', True) else '❌ NO'}")

        if "message" in result:
            print(f"   Message: {result['message'][:100]}...")

        self.test_count += 1
        if result.get("success", False):
            self.passed_count += 1

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📈 TEST SUMMARY")
        print("=" * 60)

        print(f"\nTests run: {self.test_count}")
        print(f"Tests passed: {self.passed_count}")
        print(f"Success rate: {self.passed_count / max(1, self.test_count) * 100:.1f}%")

        print("\n📋 Agent status:")
        for agent_name, result in self.results.items():
            status = "✅ PASS" if result and result.get("success", False) else "❌ FAIL"
            confidence = result.get("confidence", 0) if result else 0
            print(f"  {agent_name:20} {status:10} Confidence: {confidence:.1%}")


async def run_all_tests():
    """Run all agent tests"""
    tester = AgentTester()

    print("🤖 AI RESEARCH ASSISTANT - AGENT TEST SUITE")
    print("=" * 60)

    # Run individual agent tests
    await tester.test_domain_scout()
    await asyncio.sleep(1)  # Brief pause between tests

    await tester.test_question_generator()
    await asyncio.sleep(1)

    await tester.test_data_alchemist()
    await asyncio.sleep(1)

    await tester.test_experiment_designer()
    await asyncio.sleep(1)

    await tester.test_critic()
    await asyncio.sleep(1)

    await tester.test_uncertainty_agent()
    await asyncio.sleep(1)

    await tester.test_orchestrator()
    await asyncio.sleep(1)

    # Run integration test
    await tester.test_integration()

    # Print summary
    tester.print_summary()

    # Save results to file
    save_results(tester.results)

    return tester.passed_count == tester.test_count


def save_results(results: Dict):
    """Save test results to JSON file"""
    try:
        # Convert results to serializable format
        serializable = {}
        for agent, result in results.items():
            if result:
                # Remove any non-serializable objects
                serializable[agent] = {
                    k: v for k, v in result.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }

        filename = f"agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")


def quick_test():
    """Quick test for basic functionality"""
    print("🔧 QUICK FUNCTIONALITY TEST")
    print("=" * 60)

    # Test imports
    try:
        from agents.base_agent import BaseAgent
        print("✅ BaseAgent import: SUCCESS")
    except Exception as e:
        print(f"❌ BaseAgent import: FAILED - {e}")
        return False

    # Test agent creation
    try:
        from agents.domain_scout import DomainScoutAgent
        agent = DomainScoutAgent()
        print(f"✅ DomainScoutAgent creation: SUCCESS")
        print(f"   Agent name: {agent.name}")
        print(f"   Agent description: {agent.description}")
        return True
    except Exception as e:
        print(f"❌ DomainScoutAgent creation: FAILED - {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AI Research Assistant Agents")
    parser.add_argument("--quick", action="store_true", help="Run quick functionality test only")
    parser.add_argument("--agent", type=str,
                        help="Test specific agent (domain, question, data, experiment, critic, uncertainty, orchestrator)")

    args = parser.parse_args()

    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)

    elif args.agent:
        # Test specific agent
        tester = AgentTester()

        agent_tests = {
            "domain": tester.test_domain_scout,
            "question": tester.test_question_generator,
            "data": tester.test_data_alchemist,
            "experiment": tester.test_experiment_designer,
            "critic": tester.test_critic,
            "uncertainty": tester.test_uncertainty_agent,
            "orchestrator": tester.test_orchestrator
        }

        if args.agent in agent_tests:
            print(f"🧪 Testing {args.agent} agent only")
            asyncio.run(agent_tests[args.agent]())
            tester.print_summary()
        else:
            print(f"❌ Unknown agent: {args.agent}")
            print(f"Available agents: {', '.join(agent_tests.keys())}")
            sys.exit(1)
    else:
        # Run all tests
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)