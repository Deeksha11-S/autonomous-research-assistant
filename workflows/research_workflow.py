"""
Module
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from utils.config import Config
import logging

# Create module-level logger that will be setup later
logger = logging.getLogger(__name__)

# Try different import styles based on langgraph version
try:
    # For newer versions (0.1.0+)
    from langgraph.graph import StateGraph, END

    print("✅ Using langgraph")

    # Try to import MemorySaver, fallback to custom if not available
    try:
        from langgraph.checkpoint import MemorySaver

        print("✅ Using MemorySaver from langgraph.checkpoint")
    except ImportError:
        # Create custom MemorySaver for older versions
        class MemorySaver:
            def __init__(self):
                self.memory = {}

            def get(self, config):
                thread_id = config.get("thread_id", "default")
                return self.memory.get(thread_id, {})

            def put(self, config, value):
                thread_id = config.get("thread_id", "default")
                self.memory[thread_id] = value

            def update(self, config, key, value):
                thread_id = config.get("thread_id", "default")
                if thread_id not in self.memory:
                    self.memory[thread_id] = {}
                self.memory[thread_id][key] = value

            def list(self, config):
                thread_id = config.get("thread_id", "default")
                return list(self.memory.get(thread_id, {}).keys())


        print("✅ Using custom MemorySaver")

except ImportError as e:
    print(f"⚠️ LangGraph import issue: {e}")


    # Create minimal replacements for testing
    class StateGraph:
        def __init__(self, state):
            self.state = state
            self.nodes = {}
            self.edges = {}

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, start, end):
            self.edges[(start, end)] = True

        def set_entry_point(self, node):
            self.entry = node

        def compile(self, **kwargs):
            return self

        async def ainvoke(self, state, config):
            return state


    END = "END"


    class MemorySaver:
        def __init__(self):
            self.memory = {}

        def get(self, config):
            return self.memory.get(config.get("thread_id", "default"), {})

        def put(self, config, value):
            thread_id = config.get("thread_id", "default")
            self.memory[thread_id] = value

try:
    from langchain_core.messages import HumanMessage, AIMessage

    print("✅ Using langchain_core.messages")
except ImportError:
    # Simple replacements for testing
    class HumanMessage:
        def __init__(self, content=""):
            self.content = content


    class AIMessage:
        def __init__(self, content=""):
            self.content = content


class ResearchWorkflow:
    def __init__(self):
        self.workflow = None
        self.checkpointer = MemorySaver()
        self._build_workflow()

    def _build_workflow(self):
        """Build the LangGraph workflow"""

        # Define state - using simple dict instead of TypedDict for compatibility
        # Define the state structure
        from typing import TypedDict, Annotated
        import operator

        # Define state type
        class AgentState(TypedDict):
            messages: Annotated[List, operator.add]
            iteration: int
            domain: Optional[str]
            selected_question: Optional[str]
            data_sources: List[Dict]
            insights: List[str]
            hypothesis: Optional[Dict]
            experiments: List[Dict]
            critique: Optional[Dict]
            needs_iteration: bool
            research_paper: Optional[str]
            confidence_scores: Dict[str, float]

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes (these would be connected to actual agent methods)
        workflow.add_node("domain_scout", self._domain_scout_node)
        workflow.add_node("question_generator", self._question_generator_node)
        workflow.add_node("data_alchemist", self._data_alchemist_node)
        workflow.add_node("experiment_designer", self._experiment_designer_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("uncertainty", self._uncertainty_node)
        workflow.add_node("compile_results", self._compile_results_node)

        # Define edges
        workflow.set_entry_point("domain_scout")

        workflow.add_edge("domain_scout", "question_generator")
        workflow.add_edge("question_generator", "data_alchemist")
        workflow.add_edge("data_alchemist", "experiment_designer")
        workflow.add_edge("experiment_designer", "critic")
        workflow.add_edge("critic", "uncertainty")

        # Conditional edge from uncertainty
        workflow.add_conditional_edges(
            "uncertainty",
            self._should_continue,
            {
                "iterate": "domain_scout",
                "compile": "compile_results"
            }
        )

        workflow.add_edge("compile_results", END)

        # Compile workflow - handle different compile signatures
        try:
            self.workflow = workflow.compile(
                checkpointer=self.checkpointer,
                interrupt_before=["critic"]  # Allow human intervention if needed
            )
        except TypeError:
            # Fallback for older versions
            try:
                self.workflow = workflow.compile(checkpointer=self.checkpointer)
            except:
                self.workflow = workflow.compile()

        print("✅ Workflow built successfully")

    async def _domain_scout_node(self, state: dict) -> dict:
        """Domain scout node implementation"""
        logger.info("Running Domain Scout agent...")

        # Lazy import to avoid circular imports
        try:
            from agents.domain_scout import DomainScoutAgent
            agent = DomainScoutAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"DomainScoutAgent not available: {e}")
            result = {"domains": ["test_domain"]}

        state["messages"].append(
            HumanMessage(content=f"Find emerging domains in {state.get('domain', 'science')}")
        )
        state["messages"].append(
            AIMessage(content=f"Found domains: {result.get('domains', [])}")
        )

        if result.get("domains"):
            state["domain"] = result["domains"][0]

        return state

    async def _question_generator_node(self, state: dict) -> dict:
        """Question generator node implementation"""
        logger.info("Running Question Generator agent...")

        try:
            from agents.question_generator import QuestionGeneratorAgent
            agent = QuestionGeneratorAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"QuestionGeneratorAgent not available: {e}")
            result = {"questions": [{"question": "What are the key factors?"}]}

        state["messages"].append(
            HumanMessage(content=f"Generate questions for domain: {state.get('domain')}")
        )

        if result.get("questions"):
            state["selected_question"] = result["questions"][0].get("question", "")
            state["messages"].append(
                AIMessage(content=f"Generated question: {state['selected_question']}")
            )

        return state

    async def _data_alchemist_node(self, state: dict) -> dict:
        """Data alchemist node implementation"""
        logger.info("Running Data Alchemist agent...")

        try:
            from agents.data_alchemist import DataAlchemistAgent
            agent = DataAlchemistAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"DataAlchemistAgent not available: {e}")
            result = {"data_sources": [{"name": "test_source"}], "insights": ["test_insight"]}

        state["data_sources"] = result.get("data_sources", [])
        state["insights"] = result.get("insights", [])

        state["messages"].append(
            AIMessage(
                content=f"Gathered {len(state['data_sources'])} data sources and {len(state['insights'])} insights")
        )

        return state

    async def _experiment_designer_node(self, state: dict) -> dict:
        """Experiment designer node implementation"""
        logger.info("Running Experiment Designer agent...")

        try:
            from agents.experiment_designer import ExperimentDesignerAgent
            agent = ExperimentDesignerAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"ExperimentDesignerAgent not available: {e}")
            result = {"hypothesis": {"test": "hypothesis"}, "experiments": [{"name": "test_experiment"}]}

        state["hypothesis"] = result.get("hypothesis", {})
        state["experiments"] = result.get("experiments", [])

        state["messages"].append(
            AIMessage(content=f"Designed {len(state['experiments'])} experiments")
        )

        return state

    async def _critic_node(self, state: dict) -> dict:
        """Critic node implementation"""
        logger.info("Running Critic agent...")

        try:
            from agents.critic import CriticAgent
            agent = CriticAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"CriticAgent not available: {e}")
            result = {"critique": {"notes": "test critique"}, "needs_iteration": False}

        state["critique"] = result.get("critique", {})
        state["needs_iteration"] = result.get("needs_iteration", False)

        state["messages"].append(
            AIMessage(content=f"Critique complete: {'Needs iteration' if state['needs_iteration'] else 'Approved'}")
        )

        return state

    async def _uncertainty_node(self, state: dict) -> dict:
        """Uncertainty quantification node"""
        logger.info("Running Uncertainty agent...")

        try:
            from agents.uncertainty_agent import UncertaintyAgent
            agent = UncertaintyAgent()
            result = await agent.execute(state)
        except ImportError as e:
            logger.warning(f"UncertaintyAgent not available: {e}")
            result = {"overall_confidence": 0.8}

        # Initialize confidence_scores if not exists
        if "confidence_scores" not in state:
            state["confidence_scores"] = {}

        iteration_key = f"iteration_{state.get('iteration', 1)}"
        state["confidence_scores"][iteration_key] = result.get("overall_confidence", 0.0)

        state["messages"].append(
            AIMessage(content=f"Uncertainty assessment: {result.get('overall_confidence', 0.0):.1%} confidence")
        )

        return state

    async def _compile_results_node(self, state: dict) -> dict:
        """Compile final results node"""
        logger.info("Compiling research results...")

        # Use orchestrator to compile paper
        try:
            from agents.orchestrator import OrchestratorAgent
            orchestrator = OrchestratorAgent()
            research_paper = await orchestrator._compile_research_paper(state)
        except (ImportError, AttributeError) as e:
            logger.warning(f"OrchestratorAgent not available: {e}")
            research_paper = "# Research Paper\n\nTest content"

        state["research_paper"] = research_paper

        state["messages"].append(
            AIMessage(content="Research paper compiled successfully")
        )

        return state

    def _should_continue(self, state: dict) -> str:
        """Determine whether to iterate or compile results"""
        iteration = state.get("iteration", 1)
        needs_iteration = state.get("needs_iteration", False)

        # Get max iterations from config or default
        try:
            max_iterations = Config.MAX_ITERATIONS
        except:
            max_iterations = 3  # Default

        if iteration >= max_iterations:
            return "compile"

        if needs_iteration:
            # Increment iteration counter
            state["iteration"] = iteration + 1
            return "iterate"

        return "compile"

    async def run(self, initial_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the complete workflow"""
        if initial_state is None:
            initial_state = {
                "messages": [],
                "iteration": 1,
                "domain": None,
                "selected_question": None,
                "data_sources": [],
                "insights": [],
                "hypothesis": None,
                "experiments": [],
                "critique": None,
                "needs_iteration": False,
                "research_paper": None,
                "confidence_scores": {}
            }

        try:
            logger.info("Starting research workflow...")

            # Run the workflow
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": "research_thread_1"}}
            )

            logger.info("Workflow completed successfully")
            return final_state

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {"error": str(e), "state": initial_state}

    def get_workflow_graph(self) -> str:
        """Get visual representation of workflow"""
        try:
            return self.workflow.get_graph().draw_mermaid()
        except:
            return "Workflow graph not available"