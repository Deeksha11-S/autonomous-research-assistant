"""
Uncertainty Agent: Quantifies confidence and uncertainty
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class UncertaintyAgent(BaseAgent):
    def __init__(self):
        super().__init__("UncertaintyAgent", "Quantify confidence and uncertainty")
        self.confidence_threshold = 0.6
        self.calibration_samples = 100

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.name} quantifying uncertainty...")

        try:
            iteration = context.get("iteration", 1)
            all_components = {
                "domain": context.get("domain_data", {}),
                "question": context.get("question_data", {}),
                "data": context.get("data_sources", []),
                "hypothesis": context.get("hypothesis", {}),
                "experiments": context.get("experiments", []),
                "critique": context.get("critique_results", {}),
                "results": context.get("experiment_results", [])
            }

            component_uncertainties = await self._quantify_component_uncertainties(all_components)
            calibrated_uncertainties = await self._calibrate_uncertainties(component_uncertainties)
            overall_confidence, confidence_interval = await self._calculate_overall_confidence(calibrated_uncertainties)
            critical_sources = await self._identify_critical_uncertainties(calibrated_uncertainties)
            should_abstain = overall_confidence < self.confidence_threshold
            confidence_metrics = await self._calculate_confidence_metrics(
                calibrated_uncertainties,
                overall_confidence,
                confidence_interval
            )

            self.confidence = overall_confidence
            uncertainty_report = await self._generate_uncertainty_report(
                calibrated_uncertainties,
                overall_confidence,
                confidence_interval,
                critical_sources,
                should_abstain,
                confidence_metrics
            )

            return self.format_response(
                success=True,
                data={
                    "component_uncertainties": calibrated_uncertainties,
                    "overall_confidence": overall_confidence,
                    "confidence_interval": confidence_interval,
                    "critical_uncertainty_sources": critical_sources,
                    "should_abstain": should_abstain,
                    "confidence_metrics": confidence_metrics,
                    "uncertainty_report": uncertainty_report,
                    "assessment_metrics": {
                        "components_assessed": len(calibrated_uncertainties),
                        "confidence_threshold": self.confidence_threshold,
                        "threshold_met": overall_confidence >= self.confidence_threshold
                    }
                },
                message=f"Overall confidence: {overall_confidence:.1%} {'(ABSTAIN)' if should_abstain else '(PROCEED)'}"
            )

        except Exception as e:
            logger.error(f"Uncertainty quantification failed: {e}")
            return self.format_response(
                success=False,
                data={
                    "overall_confidence": 0.3,
                    "should_abstain": True
                },
                message=f"Uncertainty assessment failed: {str(e)}"
            )

    async def _quantify_component_uncertainties(self, components: Dict[str, Any]) -> Dict[str, Dict]:
        uncertainties = {}

        if components.get("domain"):
            uncertainties["domain"] = await self._assess_domain_uncertainty(components["domain"])

        if components.get("question"):
            uncertainties["question"] = await self._assess_question_uncertainty(components["question"])

        if components.get("data"):
            uncertainties["data"] = await self._assess_data_uncertainty(components["data"])

        if components.get("hypothesis"):
            uncertainties["hypothesis"] = await self._assess_hypothesis_uncertainty(components["hypothesis"])

        if components.get("experiments"):
            uncertainties["experiment"] = await self._assess_experiment_uncertainty(components["experiments"])

        if components.get("critique"):
            uncertainties["critique"] = await self._assess_critique_uncertainty(components["critique"])

        if components.get("results"):
            uncertainties["results"] = await self._assess_results_uncertainty(components["results"])

        return uncertainties

    async def _assess_domain_uncertainty(self, domain_data: Dict) -> Dict:
        assessment_prompt = f"""Assess uncertainty in this domain selection:

DOMAIN DATA: {json.dumps(domain_data, indent=2)}

Return JSON:
{{
    "overall_confidence": 0-1,
    "uncertainty_sources": ["source1", "source2"],
    "assessment": "assessment text"
}}"""

        try:
            response = await self.llm.generate(assessment_prompt, task_type="analysis")
            content = response.get("content", '{"overall_confidence": 0.5}')
            confidence = response.get("confidence_score", 0.7)

            try:
                assessment = json.loads(content)
                confidence = assessment.get("overall_confidence", confidence)
            except json.JSONDecodeError:
                assessment = {"overall_confidence": confidence}

            return {
                "confidence": confidence,
                "uncertainty": 1 - confidence,
                "sources": assessment.get("uncertainty_sources", ["domain_definition"]),
                "assessment": assessment.get("assessment", "Domain uncertainty moderate"),
                "llm_confidence": confidence
            }

        except Exception as e:
            logger.warning(f"Domain uncertainty assessment failed: {e}")
            return {
                "confidence": 0.3,
                "uncertainty": 0.7,
                "sources": ["assessment_failed"],
                "assessment": "Automatic assessment failed"
            }

    async def _assess_question_uncertainty(self, question_data: Dict) -> Dict:
        assessment_prompt = f"""Assess uncertainty in this research question:

QUESTION DATA: {json.dumps(question_data, indent=2)}

Return JSON with confidence score."""

        try:
            response = await self.llm.generate(assessment_prompt, task_type="analysis")
            content = response.get("content", '{"confidence": 0.6}')
            confidence = response.get("confidence_score", 0.6)

            try:
                assessment = json.loads(content)
                confidence = assessment.get("confidence", confidence)
            except:
                pass

            return {
                "confidence": confidence,
                "uncertainty": 1 - confidence,
                "sources": ["feasibility", "specificity"],
                "assessment": "Question uncertainty assessed"
            }

        except Exception as e:
            logger.warning(f"Question uncertainty assessment failed: {e}")
            return {
                "confidence": 0.3,
                "uncertainty": 0.7,
                "sources": ["assessment_failed"],
                "assessment": "Assessment failed"
            }

    async def _assess_data_uncertainty(self, data_sources: List[Dict]) -> Dict:
        if not data_sources:
            return {
                "confidence": 0.1,
                "uncertainty": 0.9,
                "sources": ["no_data"],
                "assessment": "No data available"
            }

        source_count = len(data_sources)
        source_types = len(set(s.get("type", "") for s in data_sources))

        confidence = 0.5
        if source_count >= 5:
            confidence += 0.2
        elif source_count >= 3:
            confidence += 0.1

        if source_types >= 3:
            confidence += 0.15
        elif source_types >= 2:
            confidence += 0.05

        confidence = max(0.1, min(0.95, confidence))

        return {
            "confidence": confidence,
            "uncertainty": 1 - confidence,
            "sources": ["data_quality", "source_diversity"],
            "assessment": f"Data from {source_count} sources, {source_types} types",
            "metrics": {
                "source_count": source_count,
                "source_diversity": source_types
            }
        }

    async def _assess_hypothesis_uncertainty(self, hypothesis: Dict) -> Dict:
        if not hypothesis:
            return {
                "confidence": 0.2,
                "uncertainty": 0.8,
                "sources": ["no_hypothesis"],
                "assessment": "No hypothesis formulated"
            }

        assessment_prompt = f"""Assess uncertainty in this hypothesis:

HYPOTHESIS: {json.dumps(hypothesis, indent=2)}

Return confidence assessment."""

        try:
            response = await self.llm.generate(assessment_prompt, task_type="analysis")
            content = response.get("content", '{"confidence": 0.6}')
            confidence = response.get("confidence_score", 0.6)

            try:
                assessment = json.loads(content)
                confidence = assessment.get("confidence", confidence)
            except:
                confidence = hypothesis.get("testability_score", 5) / 10

            return {
                "confidence": confidence,
                "uncertainty": 1 - confidence,
                "sources": ["testability", "assumptions"],
                "assessment": f"Hypothesis confidence: {confidence:.1%}"
            }

        except Exception as e:
            logger.warning(f"Hypothesis uncertainty assessment failed: {e}")
            testability = hypothesis.get("testability_score", 5) / 10
            return {
                "confidence": testability,
                "uncertainty": 1 - testability,
                "sources": ["assessment_failed"],
                "assessment": f"Testability score: {testability:.1%}"
            }

    async def _assess_experiment_uncertainty(self, experiments: List[Dict]) -> Dict:
        if not experiments:
            return {
                "confidence": 0.1,
                "uncertainty": 0.9,
                "sources": ["no_experiments"],
                "assessment": "No experiments designed"
            }

        valid_experiments = sum(1 for exp in experiments if exp.get("validation", {}).get("is_valid", False))
        total_experiments = len(experiments)

        confidence = 0.5
        if total_experiments > 0:
            validation_ratio = valid_experiments / total_experiments
            confidence += (validation_ratio - 0.5) * 0.3

        confidence = max(0.1, min(0.95, confidence))

        return {
            "confidence": confidence,
            "uncertainty": 1 - confidence,
            "sources": ["design_validity", "methodology"],
            "assessment": f"{valid_experiments}/{total_experiments} valid experiments"
        }

    async def _assess_critique_uncertainty(self, critique_results: Dict) -> Dict:
        if not critique_results:
            return {
                "confidence": 0.3,
                "uncertainty": 0.7,
                "sources": ["no_critique"],
                "assessment": "No critique performed"
            }

        needs_iteration = critique_results.get("needs_iteration", True)
        issue_count = critique_results.get("total_issues", 0)

        base_confidence = 0.7
        if issue_count > 0:
            base_confidence += min(0.2, issue_count * 0.05)

        if not needs_iteration:
            base_confidence += 0.15

        confidence = max(0.3, min(0.95, base_confidence))

        return {
            "confidence": confidence,
            "uncertainty": 1 - confidence,
            "sources": ["critique_thoroughness"],
            "assessment": f"Critique found {issue_count} issues"
        }

    async def _assess_results_uncertainty(self, results: List[Dict]) -> Dict:
        if not results:
            return {
                "confidence": 0.2,
                "uncertainty": 0.8,
                "sources": ["no_results"],
                "assessment": "No experimental results"
            }

        p_values = [r.get("p_value") for r in results if r.get("p_value") is not None]
        effect_sizes = [r.get("effect_size") for r in results if r.get("effect_size") is not None]

        confidence = 0.5

        if p_values:
            sig_ratio = sum(1 for p in p_values if p < 0.05) / len(p_values)
            confidence += (sig_ratio - 0.5) * 0.2

        if effect_sizes:
            avg_effect = np.mean([abs(e) for e in effect_sizes]) if effect_sizes else 0
            if avg_effect >= 0.5:
                confidence += 0.15

        confidence = max(0.1, min(0.95, confidence))

        return {
            "confidence": confidence,
            "uncertainty": 1 - confidence,
            "sources": ["statistical_power", "effect_size"],
            "assessment": f"{len(results)} results analyzed"
        }

    async def _calibrate_uncertainties(self, uncertainties: Dict[str, Dict]) -> Dict[str, Dict]:
        calibrated = {}

        for component, assessment in uncertainties.items():
            confidence = assessment["confidence"]

            if component in ["data", "results"]:
                calibrated_confidence = confidence * 0.9 + 0.05  # Slight adjustment
            elif component in ["hypothesis", "question"]:
                calibrated_confidence = confidence * 0.8 + 0.1
            else:
                calibrated_confidence = confidence * 0.7 + 0.15

            calibrated[component] = {
                **assessment,
                "calibrated_confidence": calibrated_confidence,
                "calibrated_uncertainty": 1 - calibrated_confidence
            }

        return calibrated

    async def _calculate_overall_confidence(self, calibrated_uncertainties: Dict[str, Dict]) -> Tuple[
        float, Tuple[float, float]]:
        if not calibrated_uncertainties:
            return 0.3, (0.1, 0.5)

        weights = {
            "data": 0.25,
            "experiment": 0.20,
            "results": 0.20,
            "hypothesis": 0.15,
            "critique": 0.10,
            "question": 0.05,
            "domain": 0.05
        }

        weighted_sum = 0
        total_weight = 0

        for component, assessment in calibrated_uncertainties.items():
            weight = weights.get(component, 0.05)
            confidence = assessment.get("calibrated_confidence", assessment.get("confidence", 0.5))
            weighted_sum += confidence * weight
            total_weight += weight

        overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        confidences = [a.get("calibrated_confidence", a.get("confidence", 0.5)) for a in
                       calibrated_uncertainties.values()]

        if len(confidences) >= 3:
            std_dev = np.std(confidences)
            confidence_interval = (
                max(0, overall_confidence - 1.96 * std_dev / np.sqrt(len(confidences))),
                min(1, overall_confidence + 1.96 * std_dev / np.sqrt(len(confidences)))
            )
        else:
            confidence_interval = (overall_confidence - 0.1, overall_confidence + 0.1)

        return overall_confidence, confidence_interval

    async def _identify_critical_uncertainties(self, calibrated_uncertainties: Dict[str, Dict]) -> List[Dict]:
        critical = []

        for component, assessment in calibrated_uncertainties.items():
            confidence = assessment.get("calibrated_confidence", assessment.get("confidence", 0.5))

            if confidence < self.confidence_threshold:
                critical.append({
                    "component": component,
                    "confidence": confidence,
                    "uncertainty": 1 - confidence,
                    "sources": assessment.get("sources", ["unknown"]),
                    "impact": "critical" if confidence < 0.3 else "high" if confidence < 0.5 else "moderate"
                })

        critical.sort(key=lambda x: x["confidence"])
        return critical[:3]

    async def _calculate_confidence_metrics(self, calibrated_uncertainties: Dict[str, Dict], overall_confidence: float,
                                            confidence_interval: Tuple[float, float]) -> Dict:
        metrics = {
            "overall_confidence": overall_confidence,
            "confidence_interval": confidence_interval,
            "interval_width": confidence_interval[1] - confidence_interval[0],
            "components_assessed": len(calibrated_uncertainties),
            "components_below_threshold": sum(
                1 for a in calibrated_uncertainties.values()
                if a.get("calibrated_confidence", a.get("confidence", 0.5)) < self.confidence_threshold
            )
        }

        ci_width = metrics["interval_width"]
        if ci_width < 0.1:
            metrics["reliability"] = "high"
        elif ci_width < 0.2:
            metrics["reliability"] = "medium"
        else:
            metrics["reliability"] = "low"

        return metrics

    async def _generate_uncertainty_report(self, calibrated_uncertainties: Dict[str, Dict], overall_confidence: float,
                                           confidence_interval: Tuple[float, float], critical_sources: List[Dict],
                                           should_abstain: bool, confidence_metrics: Dict) -> str:
        report_parts = []

        report_parts.append("# 📊 UNCERTAINTY QUANTIFICATION REPORT")
        report_parts.append(f"**Overall Confidence:** {overall_confidence:.1%}")
        report_parts.append(f"**95% CI:** [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
        report_parts.append(f"**Decision:** {'🚫 ABSTAIN' if should_abstain else '✅ PROCEED'}")
        report_parts.append(f"**Threshold:** {self.confidence_threshold:.0%}")

        report_parts.append("\n## 📈 COMPONENT CONFIDENCE BREAKDOWN")

        for component, assessment in calibrated_uncertainties.items():
            confidence = assessment.get("calibrated_confidence", assessment.get("confidence", 0.5))
            status = "✅" if confidence >= self.confidence_threshold else "⚠️"

            report_parts.append(f"\n### {component.upper()} {status}")
            report_parts.append(f"- **Confidence:** {confidence:.1%}")
            report_parts.append(f"- **Assessment:** {assessment.get('assessment', 'No assessment')}")

        report_parts.append("\n## ⚠️ CRITICAL UNCERTAINTY SOURCES")

        if critical_sources:
            for i, source in enumerate(critical_sources, 1):
                report_parts.append(f"\n### {i}. {source['component'].upper()}")
                report_parts.append(f"- **Confidence:** {source['confidence']:.1%}")
                report_parts.append(f"- **Impact:** {source['impact'].upper()}")
        else:
            report_parts.append("No critical uncertainties identified.")

        report_parts.append("\n## 🎯 RECOMMENDATIONS")

        if should_abstain:
            report_parts.append("**IMMEDIATE ACTIONS REQUIRED:**")
            report_parts.append("1. Address critical uncertainties before proceeding")
            report_parts.append("2. Improve data quality and completeness")
            report_parts.append("3. Strengthen experimental design")
        else:
            report_parts.append("**RESEARCH PROCEEDS TO NEXT PHASE:**")
            report_parts.append("1. Proceed with paper compilation")
            report_parts.append("2. Document limitations transparently")

        return "\n".join(report_parts)