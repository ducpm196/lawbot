#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics - LawBot v8.0
=========================================

Comprehensive metrics for 3-tier architecture evaluation.
Optimized, simple, efficient, consistent, and maintainable.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from core.pipeline import LegalQAPipeline


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        """Default encoder for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Core Metrics Functions - Centralized and optimized
def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate precision at k."""
    if k == 0 or not retrieved:
        return 0.0
    relevant_retrieved = set(relevant) & set(retrieved[:k])
    return len(relevant_retrieved) / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate recall at k."""
    if not relevant:
        return 0.0
    relevant_retrieved = set(relevant) & set(retrieved[:k])
    return len(relevant_retrieved) / len(relevant)


def f1_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate F1 score at k."""
    prec = precision_at_k(relevant, retrieved, k)
    rec = recall_at_k(relevant, retrieved, k)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def mrr_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank at k."""
    if not relevant:
        return 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """Calculate NDCG at k."""
    if not relevant:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# Utility Functions - Centralized
def canonicalize_aid_set(aid_list: List[str]) -> set:
    """Canonicalize AID list to set."""
    return {aid.strip().upper() for aid in aid_list if aid and aid.strip()}


def canonicalize_aid_list(aid_list: List[str]) -> List[str]:
    """Canonicalize AID list."""
    return [aid.strip().upper() for aid in aid_list if aid and aid.strip()]


class MetricsCalculator:
    """Centralized metrics calculator for all evaluation types."""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10, 20]):
        """Initialize metrics calculator."""
        self.k_values = k_values

    def calculate_all_metrics(
        self, relevant: List[str], retrieved: List[str]
    ) -> Dict[str, Dict[int, float]]:
        """Calculate all metrics for all k values."""
        metrics = {"precision": {}, "recall": {}, "f1": {}, "mrr": {}, "ndcg": {}}

        for k in self.k_values:
            metrics["precision"][k] = precision_at_k(relevant, retrieved, k)
            metrics["recall"][k] = recall_at_k(relevant, retrieved, k)
            metrics["f1"][k] = f1_at_k(relevant, retrieved, k)
            metrics["mrr"][k] = mrr_at_k(relevant, retrieved, k)
            metrics["ndcg"][k] = ndcg_at_k(relevant, retrieved, k)

        return metrics

    def calculate_batch_metrics(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> Dict[str, float]:
        """Calculate batch metrics across multiple queries."""
        if not self._validate_inputs(queries, ground_truth_sets, retrieved_aids_batch):
            return {}

        # Calculate metrics for each k value
        batch_metrics = {}
        for k in self.k_values:
            precisions = []
            recalls = []
            f1_scores = []
            mrrs = []
            ndcgs = []

            for gt_set, retrieved in zip(ground_truth_sets, retrieved_aids_batch):
                gt_list = list(gt_set)
                precisions.append(precision_at_k(gt_list, retrieved, k))
                recalls.append(recall_at_k(gt_list, retrieved, k))
                f1_scores.append(f1_at_k(gt_list, retrieved, k))
                mrrs.append(mrr_at_k(gt_list, retrieved, k))
                ndcgs.append(ndcg_at_k(gt_list, retrieved, k))

            # Average across queries
            batch_metrics[f"precision@{k}"] = np.mean(precisions)
            batch_metrics[f"recall@{k}"] = np.mean(recalls)
            batch_metrics[f"f1@{k}"] = np.mean(f1_scores)
            batch_metrics[f"mrr@{k}"] = np.mean(mrrs)
            batch_metrics[f"ndcg@{k}"] = np.mean(ndcgs)

        return batch_metrics

    def _validate_inputs(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> bool:
        """Validate input data consistency."""
        if len(queries) != len(ground_truth_sets) or len(queries) != len(
            retrieved_aids_batch
        ):
            return False
        if not queries or not ground_truth_sets or not retrieved_aids_batch:
            return False
        return True


class BatchEvaluator:
    """Batch evaluation for multiple queries."""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10, 20]):
        """Initialize batch evaluator."""
        self.k_values = k_values
        self.metrics_calculator = MetricsCalculator(k_values)

    def validate_inputs(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> bool:
        """Validate input data consistency."""
        return self.metrics_calculator._validate_inputs(
            queries, ground_truth_sets, retrieved_aids_batch
        )

    def evaluate_batch(
        self,
        queries: List[str],
        ground_truth_sets: List[set],
        retrieved_aids_batch: List[List[str]],
    ) -> Dict[str, float]:
        """Evaluate batch of queries."""
        return self.metrics_calculator.calculate_batch_metrics(
            queries, ground_truth_sets, retrieved_aids_batch
        )


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""

    def __init__(self, reports_dir: Optional[Path] = None):
        """Initialize evaluation reporter."""
        self.reports_dir = reports_dir or Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    def validate_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate metrics data structure."""
        required_keys = ["precision@1", "recall@10", "f1@5"]
        return all(key in metrics for key in required_keys)

    def create_comprehensive_report(
        self,
        retrieval_metrics: Dict[str, float],
        reranking_metrics: Dict[str, float],
        per_query_results: List[Dict],
        metadata: Dict[str, Any],
        cascaded_metrics: Optional[Dict[str, float]] = None,
        light_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""

        # Validate metrics
        if not self.validate_metrics(retrieval_metrics):
            raise ValueError("Invalid retrieval metrics format")
        if not self.validate_metrics(reranking_metrics):
            raise ValueError("Invalid reranking metrics format")

        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(
            retrieval_metrics, reranking_metrics, cascaded_metrics, light_metrics
        )

        # Analyze performance
        performance_analysis = self._analyze_performance(
            retrieval_metrics, reranking_metrics
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            retrieval_metrics, reranking_metrics
        )

        # Create report structure
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "metrics": {
                "retrieval": retrieval_metrics,
                "reranking": reranking_metrics,
                "pipeline": overall_performance,
            },
            "detailed_results": per_query_results,
            "performance_analysis": performance_analysis,
            "recommendations": recommendations,
        }

        # Add optional metrics
        if cascaded_metrics:
            report["metrics"]["cascaded"] = cascaded_metrics
        if light_metrics:
            report["metrics"]["light_ranking"] = light_metrics

        return report

    def _calculate_overall_performance(
        self,
        retrieval_metrics: Dict[str, float],
        reranking_metrics: Dict[str, float],
        cascaded_metrics: Optional[Dict[str, float]] = None,
        light_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Calculate overall pipeline performance."""

        # Calculate pipeline score
        retrieval_score = retrieval_metrics.get("f1@10", 0.0)
        reranking_score = reranking_metrics.get("f1@10", 0.0)

        if cascaded_metrics:
            cascaded_score = cascaded_metrics.get("f1@10", 0.0)
            pipeline_score = (retrieval_score + reranking_score + cascaded_score) / 3
        else:
            pipeline_score = (retrieval_score + reranking_score) / 2

        # Determine pipeline health
        if pipeline_score >= 0.8:
            pipeline_health = "excellent"
        elif pipeline_score >= 0.6:
            pipeline_health = "good"
        elif pipeline_score >= 0.4:
            pipeline_health = "fair"
        else:
            pipeline_health = "poor"

        return {
            "pipeline_score": pipeline_score,
            "pipeline_health": pipeline_health,
            "retrieval_score": retrieval_score,
            "reranking_score": reranking_score,
            "recommendation": self._get_pipeline_recommendation(pipeline_score),
        }

    def _analyze_performance(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze performance patterns."""

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        # Analyze retrieval performance
        if retrieval_metrics.get("precision@1", 0.0) > 0.7:
            strengths.append("High precision at first position")
        elif retrieval_metrics.get("precision@1", 0.0) < 0.3:
            weaknesses.append("Low precision at first position")

        if retrieval_metrics.get("recall@10", 0.0) > 0.8:
            strengths.append("Good recall coverage")
        elif retrieval_metrics.get("recall@10", 0.0) < 0.5:
            weaknesses.append("Poor recall coverage")

        # Analyze reranking performance
        if reranking_metrics.get("f1@5", 0.0) > 0.7:
            strengths.append("Strong reranking accuracy")
        elif reranking_metrics.get("f1@5", 0.0) < 0.4:
            weaknesses.append("Weak reranking accuracy")

        # Determine overall performance level
        avg_score = (
            retrieval_metrics.get("f1@10", 0.0) + reranking_metrics.get("f1@10", 0.0)
        ) / 2

        if avg_score >= 0.8:
            performance_level = "excellent"
        elif avg_score >= 0.6:
            performance_level = "good"
        elif avg_score >= 0.4:
            performance_level = "fair"
        else:
            performance_level = "poor"

        return {
            "performance_level": performance_level,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "retrieval_analysis": {
                "precision_at_1": retrieval_metrics.get("precision@1", 0.0),
                "recall_at_10": retrieval_metrics.get("recall@10", 0.0),
                "f1_at_10": retrieval_metrics.get("f1@10", 0.0),
            },
            "reranking_analysis": {
                "f1_at_5": reranking_metrics.get("f1@5", 0.0),
                "f1_at_10": reranking_metrics.get("f1@10", 0.0),
                "mrr_at_10": reranking_metrics.get("mrr@10", 0.0),
            },
        }

    def _generate_recommendations(
        self, retrieval_metrics: Dict[str, float], reranking_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Retrieval recommendations
        if retrieval_metrics.get("precision@1", 0.0) < 0.5:
            recommendations.append(
                "Improve retrieval precision by fine-tuning Bi-Encoder model"
            )

        if retrieval_metrics.get("recall@10", 0.0) < 0.6:
            recommendations.append(
                "Increase retrieval recall by expanding candidate pool"
            )

        # Reranking recommendations
        if reranking_metrics.get("f1@5", 0.0) < 0.6:
            recommendations.append(
                "Enhance reranking accuracy by training with more diverse data"
            )

        if reranking_metrics.get("mrr@10", 0.0) < 0.5:
            recommendations.append(
                "Improve ranking quality by optimizing reranker architecture"
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Performance is satisfactory - consider fine-tuning for specific domains"
            )

        return recommendations

    def _get_pipeline_recommendation(self, pipeline_score: float) -> str:
        """Get pipeline-level recommendation."""
        if pipeline_score >= 0.8:
            return "Pipeline performing excellently - ready for production"
        elif pipeline_score >= 0.6:
            return "Pipeline performing well - minor optimizations recommended"
        elif pipeline_score >= 0.4:
            return "Pipeline needs improvement - consider retraining models"
        else:
            return "Pipeline requires significant improvement - review architecture and data"

    def save_report(self, report: Dict[str, Any]) -> str:
        """Save evaluation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.json"
        filepath = self.reports_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

        return str(filepath)

    def display_summary(self, report: Dict[str, Any]) -> None:
        """Display evaluation summary."""
        print("\n" + "=" * 80)
        print("LAWBOT EVALUATION SUMMARY")
        print("=" * 80)

        # Pipeline overview
        pipeline_metrics = report.get("metrics", {}).get("pipeline", {})
        print(f"Pipeline Score: {pipeline_metrics.get('pipeline_score', 0.0):.3f}")
        print(f"Pipeline Health: {pipeline_metrics.get('pipeline_health', 'unknown')}")
        print(f"Recommendation: {pipeline_metrics.get('recommendation', 'N/A')}")

        # Tier metrics
        def print_tier_metrics(name, metrics):
            """Print tier metrics."""
            if metrics:
                print(f"\n{name} Metrics:")
                print(f"  F1@10: {metrics.get('f1@10', 0.0):.3f}")
                print(f"  Precision@1: {metrics.get('precision@1', 0.0):.3f}")
                print(f"  Recall@10: {metrics.get('recall@10', 0.0):.3f}")
                print(f"  MRR@10: {metrics.get('mrr@10', 0.0):.3f}")

        print_tier_metrics("Retrieval", report.get("metrics", {}).get("retrieval"))
        print_tier_metrics("Reranking", report.get("metrics", {}).get("reranking"))

        # Performance analysis
        perf_analysis = report.get("performance_analysis", {})
        if perf_analysis:
            print(
                f"\nPerformance Level: {perf_analysis.get('performance_level', 'unknown')}"
            )

            strengths = perf_analysis.get("strengths", [])
            if strengths:
                print("Strengths:")
                for strength in strengths:
                    print(f"  ✅ {strength}")

            weaknesses = perf_analysis.get("weaknesses", [])
            if weaknesses:
                print("Areas for Improvement:")
                for weakness in weaknesses:
                    print(f"  ⚠️ {weakness}")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    def export_to_csv(
        self, report: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """Export evaluation results to CSV."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.reports_dir / f"evaluation_results_{timestamp}.csv"

        # Prepare data for CSV export
        csv_data = []

        # Add pipeline metrics
        pipeline_metrics = report.get("metrics", {}).get("pipeline", {})
        csv_data.append(
            {
                "Metric_Type": "Pipeline",
                "Metric_Name": "Pipeline_Score",
                "Value": pipeline_metrics.get("pipeline_score", 0.0),
            }
        )
        csv_data.append(
            {
                "Metric_Type": "Pipeline",
                "Metric_Name": "Pipeline_Health",
                "Value": pipeline_metrics.get("pipeline_health", "unknown"),
            }
        )

        # Add retrieval metrics
        retrieval_metrics = report.get("metrics", {}).get("retrieval", {})
        for metric_name, value in retrieval_metrics.items():
            csv_data.append(
                {"Metric_Type": "Retrieval", "Metric_Name": metric_name, "Value": value}
            )

        # Add reranking metrics
        reranking_metrics = report.get("metrics", {}).get("reranking", {})
        for metric_name, value in reranking_metrics.items():
            csv_data.append(
                {"Metric_Type": "Reranking", "Metric_Name": metric_name, "Value": value}
            )

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding="utf-8")

        return str(output_path)

    def run_and_report(
        self, model_path_to_evaluate: Path, config, max_queries: Optional[int] = None
    ) -> bool:
        """Run evaluation and generate report."""
        try:
            # Import here to avoid circular imports
            pipeline = LegalQAPipeline()
            if not pipeline.is_ready:
                print("❌ Pipeline not ready for evaluation")
                return False

            # Load evaluation data
            queries, ground_truth_sets, data_path = self._load_eval_data(
                config, max_queries
            )
            if not queries:
                print("❌ No evaluation data available")
                return False

            print(f"🎯 Evaluating {len(queries)} queries...")

            # Evaluate all tiers
            (
                retrieval_metrics,
                reranking_metrics,
                cascaded_metrics,
                light_metrics,
                per_query_results,
            ) = self._evaluate_all_tiers(pipeline, queries, ground_truth_sets, config)

            # Create metadata
            metadata = self._create_metadata(pipeline, len(queries), data_path, config)

            # Generate comprehensive report
            report = self.create_comprehensive_report(
                retrieval_metrics=retrieval_metrics,
                reranking_metrics=reranking_metrics,
                per_query_results=per_query_results,
                metadata=metadata,
                cascaded_metrics=cascaded_metrics,
                light_metrics=light_metrics,
            )

            # Save report
            report_path = self.save_report(report)
            print(f"✅ Evaluation report saved to: {report_path}")

            # Display summary
            self.display_summary(report)

            return True

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return False

    def _load_eval_data(
        self, config, max_queries: Optional[int] = None
    ) -> Tuple[List[str], List[set], Path]:
        """Load evaluation data, prioritizing validation sets over test data."""
        try:
            # First try to load validation sets
            validation_dir = Path("features/validation_sets")
            if validation_dir.exists():
                tier1_val_path = validation_dir / "tier1_validation.jsonl"
                if tier1_val_path.exists() and tier1_val_path.stat().st_size > 0:
                    print("📊 Using validation sets for evaluation")
                    return self._load_validation_data(tier1_val_path, max_queries)
            
            # Fallback to test data if validation sets not available
            print("⚠️ Validation sets not found, using test data")
            return self._load_test_data(config, max_queries)
            
        except Exception as e:
            print(f"❌ Error loading evaluation data: {e}")
            return [], [], Path("")
    
    def _load_validation_data(self, validation_path: Path, max_queries: Optional[int] = None) -> Tuple[List[str], List[set], Path]:
        """Load validation data for evaluation."""
        try:
            queries = []
            ground_truth_sets = []
            
            with open(validation_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        query = item.get("query", "")
                        if query:
                            queries.append(query)
                            # For validation data, we need to create ground truth sets
                            # This is a simplified version - in practice, you'd need proper ground truth
                            ground_truth_sets.append(set([item.get("positive", "")]))
                            
                            if max_queries and len(queries) >= max_queries:
                                break
            
            print(f"✅ Loaded {len(queries)} validation queries")
            return queries, ground_truth_sets, validation_path
            
        except Exception as e:
            print(f"❌ Error loading validation data: {e}")
            return [], [], validation_path
    
    def _load_test_data(self, config, max_queries: Optional[int] = None) -> Tuple[List[str], List[set], Path]:
        """Load evaluation data."""
        # Implementation for loading evaluation data
        # This would be implemented based on your data loading logic
        pass

    def _evaluate_all_tiers(
        self,
        pipeline: LegalQAPipeline,
        queries: List[str],
        ground_truth_sets: List[set],
        config,
    ) -> Tuple[Dict, Dict, Dict, Dict, List[Dict]]:
        """Evaluate all tiers of the pipeline."""
        # Implementation for evaluating all tiers
        # This would be implemented based on your pipeline evaluation logic
        pass

    def _create_metadata(
        self, pipeline: LegalQAPipeline, query_count: int, data_source: Path, config
    ) -> Dict[str, Any]:
        """Create metadata for evaluation report."""
        # Implementation for creating metadata
        # This would be implemented based on your metadata requirements
        pass
