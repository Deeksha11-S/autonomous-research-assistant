"""
Critic Agent: Critiques methodology and forces iterations
"""

import asyncio
import json
from typing import Dict, Any, List, Tuple
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__("Critic", "Critique methodology and force iterations")
        self.p_value_threshold = 0.05
        self.min_effect_size = 0.2
        self.max_iterations = 5

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.name} critiquing research...")

        try:
            iteration = context.get("iteration", 1)
            hypothesis = context.get("hypothesis", {})
            experiments = context.get("experiments", [])
            experiment_results = context.get("experiment_results", [])

            if not experiments:
                return self.format_response(
                    success=False,
                    data={"critique": {}, "needs_iteration": True},
                    message="No experiments to critique"
                )

            methodology_critique = await self._critique_methodology(experiments)
            statistical_critique = await self._critique_statistics(experiment_results)
            assumption_critique = await self._critique_assumptions(hypothesis, experiments)

            needs_iteration, iteration_reasons = await self._determine_iteration_needed(
                methodology_critique,
                statistical_critique,
                assumption_critique,
                iteration
            )

            critique_severity = self._calculate_critique_severity(
                methodology_critique,
                statistical_critique,
                assumption_critique
            )

            self.confidence = self.calculate_confidence({
                "completeness": 1.0,
                "validation_passed": not needs_iteration,
                "data_quality": 1.0 - (critique_severity * 0.1)
            })

            feedback = await self._generate_constructive_feedback(
                methodology_critique,
                statistical_critique,
                assumption_critique,
                needs_iteration,
                iteration_reasons
            )

            return self.format_response(
                success=True,
                data={
                    "methodology_critique": methodology_critique,
                    "statistical_critique": statistical_critique,
                    "assumption_critique": assumption_critique,
                    "iteration_decision": {
                        "needs_iteration": needs_iteration,
                        "reasons": iteration_reasons,
                        "current_iteration": iteration,
                        "max_iterations": self.max_iterations
                    },
                    "feedback": feedback,
                    "critique_metrics": {
                        "total_issues": len(methodology_critique.get("issues", [])) +
                                       len(statistical_critique.get("issues", [])) +
                                       len(assumption_critique.get("issues", [])),
                        "severity_level": critique_severity
                    }
                },
                message=f"Critique complete: {'Needs iteration' if needs_iteration else 'Approved'}"
            )

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return self.format_response(
                success=False,
                data={"critique": {}, "needs_iteration": True},
                message=f"Critique failed: {str(e)}"
            )

    async def _critique_methodology(self, experiments: List[Dict]) -> Dict:
        critique_prompt = f"""Critique the methodology of these experiments:

EXPERIMENTS: {json.dumps(experiments, indent=2)}

Return JSON critique report:
{{
    "issues": ["issue1", "issue2"],
    "severity": "low/medium/high/critical",
    "summary": "summary text"
}}"""

        try:
            response = await self.llm.generate(critique_prompt, task_type="analysis")
            content = response.get("content", '{"issues": [], "severity": "low"}')
            try:
                critique = json.loads(content)
            except json.JSONDecodeError:
                critique = {"issues": ["Methodology critique failed"], "severity": "medium"}

            return critique

        except Exception as e:
            logger.warning(f"Methodology critique failed: {e}")
            return {"issues": ["Methodology critique failed"], "severity": "medium"}

    async def _critique_statistics(self, experiment_results: List[Dict]) -> Dict:
        if not experiment_results:
            return {"issues": ["No experimental results"], "severity": "low"}

        critique_prompt = f"""Critique the statistical methods and results:

EXPERIMENT RESULTS: {json.dumps(experiment_results[:3], indent=2)}

Return JSON critique."""

        try:
            response = await self.llm.generate(critique_prompt, task_type="analysis")
            content = response.get("content", '{"issues": [], "severity": "low"}')
            try:
                critique = json.loads(content)
            except json.JSONDecodeError:
                critique = {"issues": ["Statistical critique failed"], "severity": "medium"}

            critique["automated_checks"] = await self._perform_statistical_checks(experiment_results)
            return critique

        except Exception as e:
            logger.warning(f"Statistical critique failed: {e}")
            return {"issues": ["Statistical critique failed"], "severity": "medium"}

    async def _perform_statistical_checks(self, experiment_results: List[Dict]) -> List[Dict]:
        checks = []
        for result in experiment_results:
            p_value = result.get("p_value")
            if p_value is not None and p_value > self.p_value_threshold:
                checks.append({
                    "check": "p_value_significance",
                    "result": result.get("experiment_name", "Unknown"),
                    "issue": f"p-value {p_value:.4f} > {self.p_value_threshold} threshold",
                    "severity": "critical" if result.get("claimed_significant", False) else "high"
                })
        return checks

    async def _critique_assumptions(self, hypothesis: Dict, experiments: List[Dict]) -> Dict:
        critique_prompt = f"""Critique the assumptions in this research:

HYPOTHESIS: {json.dumps(hypothesis, indent=2)}
EXPERIMENTS: {len(experiments)} experiments

Return JSON critique."""

        try:
            response = await self.llm.generate(critique_prompt, task_type="analysis")
            content = response.get("content", '{"issues": []}')
            try:
                critique = json.loads(content)
            except json.JSONDecodeError:
                critique = {"issues": ["Assumption critique failed"]}

            return critique

        except Exception as e:
            logger.warning(f"Assumption critique failed: {e}")
            return {"issues": ["Assumption critique failed"]}

    async def _determine_iteration_needed(self, methodology_critique: Dict, statistical_critique: Dict, assumption_critique: Dict, current_iteration: int) -> Tuple[bool, List[str]]:
        reasons = []
        needs_iteration = False

        if current_iteration >= self.max_iterations:
            return False, ["Reached maximum iterations"]

        method_severity = methodology_critique.get("severity", "low")
        if method_severity in ["high", "critical"]:
            needs_iteration = True
            reasons.append("Critical methodology issues")

        stat_checks = statistical_critique.get("automated_checks", [])
        critical_stat_issues = [c for c in stat_checks if c.get("severity") in ["high", "critical"]]
        if critical_stat_issues:
            needs_iteration = True
            reasons.append("Critical statistical issues")

        return needs_iteration, reasons

    def _calculate_critique_severity(self, methodology_critique: Dict, statistical_critique: Dict, assumption_critique: Dict) -> int:
        severity_score = 0
        method_severity = methodology_critique.get("severity", "low")

        if method_severity == "critical":
            severity_score = 4
        elif method_severity == "high":
            severity_score = 3
        elif method_severity == "medium":
            severity_score = 2
        elif method_severity == "low":
            severity_score = 1

        return severity_score

    async def _generate_constructive_feedback(self, methodology_critique: Dict, statistical_critique: Dict, assumption_critique: Dict, needs_iteration: bool, iteration_reasons: List[str]) -> str:
        feedback_parts = []

        if needs_iteration:
            feedback_parts.append("# 🔄 ITERATION REQUIRED")
            feedback_parts.append(f"**Reasons:** {', '.join(iteration_reasons)}")
        else:
            feedback_parts.append("# ✅ APPROVED FOR FINALIZATION")

        feedback_parts.append("\n## 📋 METHODOLOGY ISSUES")
        method_issues = methodology_critique.get("issues", [])
        if method_issues:
            for i, issue in enumerate(method_issues[:3], 1):
                feedback_parts.append(f"{i}. {issue}")
        else:
            feedback_parts.append("No critical methodology issues found.")

        feedback_parts.append("\n## 📊 STATISTICAL ISSUES")
        stat_issues = statistical_critique.get("issues", [])
        if stat_issues:
            for i, issue in enumerate(stat_issues[:2], 1):
                feedback_parts.append(f"{i}. {issue}")
        else:
            feedback_parts.append("No critical statistical issues found.")

        feedback_parts.append("\n## 🎯 RECOMMENDATIONS")
        if needs_iteration:
            feedback_parts.append("1. Address critical methodology issues")
            feedback_parts.append("2. Improve statistical power")
            feedback_parts.append("3. Gather additional data if needed")
        else:
            feedback_parts.append("1. Proceed to final paper compilation")
            feedback_parts.append("2. Document limitations thoroughly")

        return "\n".join(feedback_parts)