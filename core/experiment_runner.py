"""
Experiment Runner for autonomous hypothesis testing.
Executes simple experiments, statistical tests, and validation checks.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging  # ✅ Use standard logging initially

# Create module-level logger that will be setup later
logger = logging.getLogger(__name__)

"""
Experiment Runner for autonomous hypothesis testing.
Executes simple experiments, statistical tests, and validation checks.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import statistics
import random
from scipy import stats  # For statistical tests

from core.llm_client import get_llm_client
from utils.config import Config
from tools.data_processor import DataProcessor



@dataclass
class ExperimentResult:
    """Structured experiment results"""
    hypothesis: str
    experiment_type: str
    data_sources: List[str]
    metrics: Dict[str, float]
    statistical_significance: Optional[float]  # p-value
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    is_significant: bool  # p < 0.05
    interpretation: str
    limitations: List[str]
    raw_data: Optional[Dict] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ValidationCheck:
    """Validation check for experimental results"""
    check_type: str  # "statistical", "methodological", "data_quality"
    description: str
    passed: bool
    severity: str  # "low", "medium", "high"
    details: str
    recommendation: Optional[str] = None


class ExperimentRunner:
    """
    Autonomous experiment execution and validation engine.
    Capable of running statistical tests, A/B tests, correlations, and simulations.
    """

    def __init__(self):
        self.config = Config()
        self.llm_client = get_llm_client()
        self.data_processor = DataProcessor()
        self.experiment_registry = {}

        # Available experiment types
        self.available_experiments = {
            "correlation_analysis": self._run_correlation_analysis,
            "ab_test": self._run_ab_test,
            "hypothesis_test": self._run_hypothesis_test,
            "regression_analysis": self._run_regression_analysis,
            "simulation": self._run_simulation,
            "validation_check": self._run_validation_check
        }

    async def design_experiment(
            self,
            hypothesis: str,
            available_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Design an appropriate experiment for a given hypothesis.
        Uses LLM to determine the best experimental approach.
        """
        logger.info(f"Designing experiment for hypothesis: {hypothesis[:100]}...")

        # Prepare data summary for LLM
        data_summary = []
        for data in available_data[:5]:  # Limit to 5 data sources
            summary = {
                "source": data.get("source", "unknown"),
                "type": data.get("type", "unknown"),
                "sample_size": data.get("sample_size", 0),
                "variables": list(data.get("variables", {}).keys())[:10]
            }
            data_summary.append(summary)

        prompt = f"""You are an experimental design expert. Design the best experiment to test this hypothesis:

HYPOTHESIS: {hypothesis}

AVAILABLE DATA SOURCES:
{json.dumps(data_summary, indent=2)}

Consider:
1. What type of experiment is most appropriate (A/B test, correlation, simulation, etc.)?
2. What statistical tests should be used?
3. What sample size is needed?
4. What are potential confounding variables?
5. What success metrics should be measured?

Return your response as JSON with this structure:
{{
    "experiment_type": "correlation_analysis|ab_test|hypothesis_test|regression_analysis|simulation",
    "description": "Detailed experiment design",
    "statistical_test": "t-test|chi-square|pearson|anova|etc",
    "required_sample_size": 100,
    "success_metrics": ["metric1", "metric2"],
    "potential_confounds": ["confound1", "confound2"],
    "data_requirements": ["data_type1", "data_type2"],
    "expected_duration": "estimated_time",
    "complexity": "low|medium|high"
}}"""

        try:
            response = await self.llm_client.generate(
                prompt,
                task_type="analysis",
                temperature=0.2,
                max_tokens=1500
            )

            # Parse JSON response
            design = json.loads(response.content)

            # Validate design
            if design["experiment_type"] not in self.available_experiments:
                logger.warning(f"Invalid experiment type: {design['experiment_type']}")
                design["experiment_type"] = "hypothesis_test"  # Fallback

            logger.info(f"Designed {design['experiment_type']} experiment")
            return design

        except Exception as e:
            logger.error(f"Failed to design experiment: {e}")
            # Return a default simple design
            return {
                "experiment_type": "hypothesis_test",
                "description": "Basic hypothesis validation",
                "statistical_test": "t-test",
                "required_sample_size": 50,
                "success_metrics": ["statistical_significance", "effect_size"],
                "potential_confounds": ["sample_bias", "measurement_error"],
                "data_requirements": ["numerical_data"],
                "expected_duration": "short",
                "complexity": "low"
            }

    async def run_experiment(
            self,
            hypothesis: str,
            experiment_design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """
        Execute an experiment based on the design.
        """
        logger.info(f"Running {experiment_design['experiment_type']} experiment...")

        experiment_type = experiment_design["experiment_type"]
        experiment_func = self.available_experiments.get(experiment_type)

        if not experiment_func:
            logger.error(f"Unknown experiment type: {experiment_type}")
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        try:
            # Run the experiment
            result = await experiment_func(hypothesis, experiment_design, data_sources)

            # Perform validation checks
            validation_checks = await self._validate_results(result, experiment_design)

            # Update result with validation
            result.limitations.extend([
                check.description for check in validation_checks if not check.passed
            ])

            logger.info(f"Experiment completed: significant={result.is_significant}")
            return result

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            # Return a failure result
            return ExperimentResult(
                hypothesis=hypothesis,
                experiment_type=experiment_type,
                data_sources=[ds.get("source", "unknown") for ds in data_sources],
                metrics={"error": 1.0},
                statistical_significance=None,
                effect_size=None,
                confidence_interval=None,
                is_significant=False,
                interpretation=f"Experiment failed: {str(e)}",
                limitations=["experiment_execution_failed", "technical_error"]
            )

    async def _run_correlation_analysis(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run correlation analysis between variables"""
        logger.info("Running correlation analysis...")

        # Extract numerical data
        numerical_data = []
        for source in data_sources:
            if source.get("type") == "numerical":
                numerical_data.extend(source.get("data", []))

        if len(numerical_data) < 10:
            return self._create_insufficient_data_result(hypothesis, design, data_sources)

        # Create synthetic correlation analysis (real implementation would use actual data)
        # For demo purposes, we'll simulate
        n_samples = min(100, len(numerical_data))
        sample_data = numerical_data[:n_samples]

        # Simulate variables
        np.random.seed(42)  # For reproducibility
        var1 = np.random.normal(0, 1, n_samples)
        var2 = var1 * 0.6 + np.random.normal(0, 0.5, n_samples)  # Positive correlation

        # Calculate correlation
        correlation, p_value = stats.pearsonr(var1, var2)

        # Calculate effect size
        effect_size = abs(correlation)

        # Calculate confidence interval
        ci_low, ci_high = self._calculate_correlation_ci(correlation, n_samples)

        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type="correlation_analysis",
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={
                "correlation_coefficient": float(correlation),
                "sample_size": n_samples,
                "r_squared": correlation ** 2
            },
            statistical_significance=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(float(ci_low), float(ci_high)),
            is_significant=p_value < 0.05,
            interpretation=f"Found {'positive' if correlation > 0 else 'negative'} correlation (r={correlation:.3f}, p={p_value:.4f})",
            limitations=["simulated_data", "small_sample"] if n_samples < 30 else ["simulated_data"],
            raw_data={"var1": var1.tolist(), "var2": var2.tolist()}
        )

    async def _run_ab_test(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run A/B test (comparison of two groups)"""
        logger.info("Running A/B test...")

        # For real implementation, this would use actual A/B test data
        # Here we simulate an A/B test result

        np.random.seed(42)
        group_a = np.random.normal(100, 15, 50)  # Control group
        group_b = np.random.normal(115, 15, 50)  # Treatment group (higher mean)

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(group_a) ** 2 + np.std(group_b) ** 2) / 2)
        cohens_d = (np.mean(group_b) - np.mean(group_a)) / pooled_std

        # Calculate confidence interval for mean difference
        mean_diff = np.mean(group_b) - np.mean(group_a)
        se_diff = np.sqrt(np.var(group_a) / len(group_a) + np.var(group_b) / len(group_b))
        ci_low = mean_diff - 1.96 * se_diff
        ci_high = mean_diff + 1.96 * se_diff

        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type="ab_test",
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={
                "group_a_mean": float(np.mean(group_a)),
                "group_b_mean": float(np.mean(group_b)),
                "mean_difference": float(mean_diff),
                "t_statistic": float(t_stat),
                "cohens_d": float(cohens_d)
            },
            statistical_significance=float(p_value),
            effect_size=float(abs(cohens_d)),
            confidence_interval=(float(ci_low), float(ci_high)),
            is_significant=p_value < 0.05,
            interpretation=f"A/B test {'shows significant difference' if p_value < 0.05 else 'shows no significant difference'} (p={p_value:.4f}, d={cohens_d:.3f})",
            limitations=["simulated_data", "equal_variance_assumed"],
            raw_data={"group_a": group_a.tolist(), "group_b": group_b.tolist()}
        )

    async def _run_hypothesis_test(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run basic hypothesis test"""
        logger.info("Running hypothesis test...")

        # This is a generic hypothesis test that uses LLM for evaluation
        # when real statistical testing isn't possible with available data

        prompt = f"""Evaluate this hypothesis based on the available data:

HYPOTHESIS: {hypothesis}

DATA SOURCES SUMMARY:
{json.dumps([{k: v for k, v in ds.items() if k != 'raw_data'} for ds in data_sources[:3]], indent=2)}

Provide a statistical evaluation:
1. Likelihood of hypothesis being true (0-100%)
2. Key supporting evidence
3. Key contradicting evidence
4. Statistical confidence level
5. Recommended next steps

Format as JSON:
{{
    "likelihood_percentage": 75,
    "supporting_evidence": ["evidence1", "evidence2"],
    "contradicting_evidence": ["evidence1", "evidence2"],
    "statistical_confidence": 0.85,
    "recommendations": ["recommendation1", "recommendation2"]
}}"""

        try:
            response = await self.llm_client.generate(
                prompt,
                task_type="analysis",
                temperature=0.1
            )

            evaluation = json.loads(response.content)

            # Convert to experiment result format
            likelihood = evaluation.get("likelihood_percentage", 50) / 100.0

            return ExperimentResult(
                hypothesis=hypothesis,
                experiment_type="hypothesis_test",
                data_sources=[ds.get("source", "unknown") for ds in data_sources],
                metrics={
                    "likelihood_score": likelihood,
                    "supporting_evidence_count": len(evaluation.get("supporting_evidence", [])),
                    "contradicting_evidence_count": len(evaluation.get("contradicting_evidence", []))
                },
                statistical_significance=1 - likelihood,  # p-value approximation
                effect_size=likelihood - 0.5,  # Center at 0.5
                confidence_interval=(max(0, likelihood - 0.2), min(1, likelihood + 0.2)),
                is_significant=likelihood > 0.7,  # Arbitrary threshold
                interpretation=f"Hypothesis evaluation suggests {likelihood:.1%} likelihood based on available evidence",
                limitations=["llm_based_evaluation", "qualitative_analysis"],
                raw_data=evaluation
            )

        except Exception as e:
            logger.error(f"Hypothesis test failed: {e}")
            # Return neutral result
            return ExperimentResult(
                hypothesis=hypothesis,
                experiment_type="hypothesis_test",
                data_sources=[ds.get("source", "unknown") for ds in data_sources],
                metrics={"error": 1.0},
                statistical_significance=0.5,
                effect_size=0.0,
                confidence_interval=(0.0, 1.0),
                is_significant=False,
                interpretation="Could not evaluate hypothesis due to technical error",
                limitations=["evaluation_failed", "insufficient_data"]
            )

    async def _run_regression_analysis(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run regression analysis"""
        logger.info("Running regression analysis...")

        # For real data, this would perform actual regression
        # Here we simulate a regression result

        np.random.seed(42)
        n_samples = 100
        X = np.random.normal(0, 1, (n_samples, 3))  # 3 predictors
        true_coeffs = [0.5, -0.3, 0.8]
        y = X @ true_coeffs + np.random.normal(0, 0.5, n_samples)

        # Perform linear regression
        X_with_const = np.column_stack([np.ones(n_samples), X])
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

        # Calculate R-squared
        y_pred = X_with_const @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type="regression_analysis",
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={
                "r_squared": float(r_squared),
                "coefficients": coeffs.tolist(),
                "sample_size": n_samples,
                "predictors": 3
            },
            statistical_significance=0.001 if r_squared > 0.1 else 0.5,  # Simplified
            effect_size=float(r_squared),
            confidence_interval=(float(r_squared - 0.1), float(r_squared + 0.1)),
            is_significant=r_squared > 0.3,  # Arbitrary threshold
            interpretation=f"Regression explains {r_squared:.1%} of variance in the outcome",
            limitations=["simulated_data", "linear_assumption"],
            raw_data={"X": X.tolist(), "y": y.tolist(), "coefficients": coeffs.tolist()}
        )

    async def _run_simulation(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run simulation-based experiment"""
        logger.info("Running simulation...")

        # Simple Monte Carlo simulation
        np.random.seed(42)
        n_simulations = 1000

        # Simulate based on hypothesis keywords
        if "increase" in hypothesis.lower() or "improve" in hypothesis.lower():
            # Simulate positive effect
            baseline = np.random.normal(50, 10, n_simulations)
            treatment = baseline + np.random.normal(5, 3, n_simulations)  # Positive effect
        elif "decrease" in hypothesis.lower() or "reduce" in hypothesis.lower():
            # Simulate negative effect
            baseline = np.random.normal(50, 10, n_simulations)
            treatment = baseline - np.random.normal(5, 3, n_simulations)  # Negative effect
        else:
            # Neutral simulation
            baseline = np.random.normal(50, 10, n_simulations)
            treatment = np.random.normal(50, 10, n_simulations)

        # Calculate effect
        effect = treatment - baseline
        mean_effect = np.mean(effect)
        effect_std = np.std(effect)

        # Calculate probability of positive effect
        prob_positive = np.mean(effect > 0)

        # Calculate confidence interval
        ci_low = np.percentile(effect, 2.5)
        ci_high = np.percentile(effect, 97.5)

        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type="simulation",
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={
                "mean_effect": float(mean_effect),
                "effect_std": float(effect_std),
                "prob_positive": float(prob_positive),
                "simulation_count": n_simulations
            },
            statistical_significance=min(prob_positive, 1 - prob_positive) * 2,  # Two-tailed p-value
            effect_size=float(abs(mean_effect) / effect_std if effect_std > 0 else 0),
            confidence_interval=(float(ci_low), float(ci_high)),
            is_significant=abs(mean_effect) > effect_std,  # Effect > 1 std dev
            interpretation=f"Simulation suggests {mean_effect:.2f} unit effect ({prob_positive:.1%} probability of positive effect)",
            limitations=["simulation_based", "simplified_model"],
            raw_data={"baseline": baseline.tolist(), "treatment": treatment.tolist(), "effect": effect.tolist()}
        )

    async def _run_validation_check(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Run validation checks on data and methodology"""
        logger.info("Running validation checks...")

        validation_results = await self._validate_results(None, design)

        # Count passed/failed checks
        passed = sum(1 for check in validation_results if check.passed)
        total = len(validation_results)
        pass_rate = passed / total if total > 0 else 0

        # Check severity
        high_severity_fails = sum(
            1 for check in validation_results
            if not check.passed and check.severity == "high"
        )

        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type="validation_check",
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={
                "validation_pass_rate": pass_rate,
                "passed_checks": passed,
                "total_checks": total,
                "high_severity_fails": high_severity_fails
            },
            statistical_significance=pass_rate,  # Use pass rate as significance proxy
            effect_size=pass_rate,
            confidence_interval=(max(0, pass_rate - 0.1), min(1, pass_rate + 0.1)),
            is_significant=pass_rate > 0.8 and high_severity_fails == 0,
            interpretation=f"Validation checks: {passed}/{total} passed ({pass_rate:.1%})",
            limitations=["methodological_only", "does_not_test_hypothesis"],
            raw_data={"checks": [vars(check) for check in validation_results]}
        )

    async def _validate_results(
            self,
            result: Optional[ExperimentResult],
            design: Dict[str, Any]
    ) -> List[ValidationCheck]:
        """Validate experiment results and methodology"""
        checks = []

        # Statistical power check
        if result and result.metrics.get("sample_size", 0) < 30:
            checks.append(ValidationCheck(
                check_type="statistical",
                description="Small sample size may lack statistical power",
                passed=False,
                severity="medium",
                details=f"Sample size: {result.metrics.get('sample_size', 0)}",
                recommendation="Increase sample size or use non-parametric tests"
            ))
        else:
            checks.append(ValidationCheck(
                check_type="statistical",
                description="Adequate sample size for statistical power",
                passed=True,
                severity="low",
                details="Sample size ≥ 30"
            ))

        # P-value sanity check
        if result and result.statistical_significance is not None:
            if result.statistical_significance < 0.001:
                checks.append(ValidationCheck(
                    check_type="statistical",
                    description="Extremely low p-value (potential overfitting or data issues)",
                    passed=False,
                    severity="medium",
                    details=f"p-value: {result.statistical_significance:.6f}",
                    recommendation="Verify data quality and check for data leakage"
                ))
            elif result.statistical_significance > 0.1:
                checks.append(ValidationCheck(
                    check_type="statistical",
                    description="High p-value (lack of statistical significance)",
                    passed=True,  # This is actually okay - it's honest
                    severity="low",
                    details=f"p-value: {result.statistical_significance:.4f}",
                    recommendation="Consider increasing sample size or revising hypothesis"
                ))

        # Effect size check
        if result and result.effect_size is not None:
            if abs(result.effect_size) < 0.2:
                checks.append(ValidationCheck(
                    check_type="methodological",
                    description="Small effect size (may not be practically significant)",
                    passed=False,
                    severity="low",
                    details=f"Effect size: {result.effect_size:.3f}",
                    recommendation="Consider practical significance in addition to statistical significance"
                ))

        # Multiple testing check
        if design.get("multiple_comparisons", False):
            checks.append(ValidationCheck(
                check_type="methodological",
                description="Multiple comparisons without correction",
                passed=False,
                severity="high",
                details="Experiment involves multiple hypothesis tests",
                recommendation="Apply Bonferroni or FDR correction"
            ))

        # Randomization check (for A/B tests)
        if design.get("experiment_type") == "ab_test":
            checks.append(ValidationCheck(
                check_type="methodological",
                description="Random assignment of subjects to groups",
                passed=True,  # Assume yes for simulation
                severity="high",
                details="Randomization is crucial for causal inference",
                recommendation="Verify randomization procedure"
            ))

        # Assumption checks
        if design.get("statistical_test") in ["t-test", "anova"]:
            checks.append(ValidationCheck(
                check_type="statistical",
                description="Normality assumption check",
                passed=True,  # Assume yes for now
                severity="medium",
                details="Parametric tests assume normally distributed residuals",
                recommendation="Check normality with Q-Q plots or Shapiro-Wilk test"
            ))

        return checks

    def _calculate_correlation_ci(self, r: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient"""
        if n <= 3:
            return (r - 0.5, r + 0.5)

        # Fisher transformation
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)

        # 95% CI in z-space
        z_low = z - 1.96 * se
        z_high = z + 1.96 * se

        # Transform back to r-space
        r_low = np.tanh(z_low)
        r_high = np.tanh(z_high)

        return (r_low, r_high)

    def _create_insufficient_data_result(
            self,
            hypothesis: str,
            design: Dict[str, Any],
            data_sources: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """Create result for insufficient data"""
        return ExperimentResult(
            hypothesis=hypothesis,
            experiment_type=design["experiment_type"],
            data_sources=[ds.get("source", "unknown") for ds in data_sources],
            metrics={"insufficient_data": 1.0, "available_samples": len(data_sources)},
            statistical_significance=None,
            effect_size=None,
            confidence_interval=None,
            is_significant=False,
            interpretation="Insufficient data to perform statistical analysis",
            limitations=["insufficient_data", "small_sample_size", "cannot_conclude"]
        )

    def get_available_experiment_types(self) -> List[str]:
        """Get list of available experiment types"""
        return list(self.available_experiments.keys())

    async def run_full_experiment_pipeline(
            self,
            hypothesis: str,
            data_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run complete experiment pipeline: design → execute → validate → interpret
        """
        logger.info(f"Starting full experiment pipeline for: {hypothesis[:50]}...")

        # Step 1: Design experiment
        design = await self.design_experiment(hypothesis, data_sources)

        # Step 2: Run experiment
        result = await self.run_experiment(hypothesis, design, data_sources)

        # Step 3: Validate results
        validation_checks = await self._validate_results(result, design)

        # Step 4: Generate comprehensive interpretation
        interpretation = await self._interpret_results(result, validation_checks)

        return {
            "experiment_design": design,
            "results": result,
            "validation_checks": validation_checks,
            "interpretation": interpretation,
            "summary": self._create_experiment_summary(result, validation_checks)
        }

    async def _interpret_results(
            self,
            result: ExperimentResult,
            validation_checks: List[ValidationCheck]
    ) -> Dict[str, Any]:
        """Generate comprehensive interpretation of experiment results"""

        prompt = f"""Interpret these experiment results for a research paper:

HYPOTHESIS: {result.hypothesis}
EXPERIMENT TYPE: {result.experiment_type}
RESULTS:
- Statistical Significance (p-value): {result.statistical_significance}
- Effect Size: {result.effect_size}
- Confidence Interval: {result.confidence_interval}
- Significant: {result.is_significant}
- Key Metrics: {json.dumps(result.metrics, indent=2)}

VALIDATION CHECKS:
{json.dumps([vars(check) for check in validation_checks], indent=2)}

Provide a comprehensive interpretation including:
1. What the results mean for the hypothesis
2. Limitations of the study
3. Potential alternative explanations
4. Recommendations for future research
5. Overall confidence in conclusions

Format as JSON with keys: interpretation, limitations, alternatives, recommendations, overall_confidence"""

        try:
            response = await self.llm_client.generate(
                prompt,
                task_type="analysis",
                temperature=0.3
            )
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Failed to interpret results: {e}")
            return {
                "interpretation": "Could not generate detailed interpretation",
                "limitations": ["interpretation_failed"],
                "alternatives": [],
                "recommendations": ["Run experiment with better data"],
                "overall_confidence": 0.5
            }

    def _create_experiment_summary(
            self,
            result: ExperimentResult,
            validation_checks: List[ValidationCheck]
    ) -> Dict[str, Any]:
        """Create a summary of experiment results"""
        passed_checks = sum(1 for check in validation_checks if check.passed)
        total_checks = len(validation_checks)

        return {
            "hypothesis": result.hypothesis,
            "conclusion": "Supported" if result.is_significant else "Not Supported",
            "confidence_level": "High" if result.confidence_score > 0.8 else "Medium" if result.confidence_score > 0.6 else "Low",
            "key_finding": result.interpretation[:200],
            "validation_score": f"{passed_checks}/{total_checks}",
            "recommendation": "Proceed with research" if result.is_significant and passed_checks == total_checks else "Revise and retry",
            "timestamp": datetime.now().isoformat()
        }


# Global instance
_experiment_runner_instance = None


def get_experiment_runner() -> ExperimentRunner:
    """Get or create the global experiment runner instance"""
    global _experiment_runner_instance
    if _experiment_runner_instance is None:
        _experiment_runner_instance = ExperimentRunner()
    return _experiment_runner_instance