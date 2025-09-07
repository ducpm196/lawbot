#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for LawBot
==========================================

This script generates comprehensive evaluation reports for the entire LawBot pipeline,
including model status, FAISS index status, and tier-specific evaluations.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.utils.system_check import get_model_status, get_faiss_index_status
    from config.loader import get_all_model_keys, get_model_status_key, get_display_name
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Please ensure the project structure is correct and dependencies are installed."
    )
    sys.exit(1)

from core.utils.logging_manager import get_logger

logger = get_logger(__name__)


def generate_comprehensive_evaluation() -> Dict[str, Any]:
    """Generate comprehensive evaluation report for the entire LawBot pipeline."""
    logger.info("🔍 Starting comprehensive evaluation...")

    try:
        # Get system status
        model_status = get_model_status()
        faiss_status = get_faiss_index_status()

        logger.info(f"📊 Model status: {len(model_status)} models checked")
        logger.info(f"📊 FAISS status: {faiss_status['status']}")

        # Generate tier evaluation
        tier_evaluation = _generate_tier_evaluation(model_status, faiss_status)

        # Generate pipeline health assessment
        pipeline_health = _calculate_pipeline_health(model_status, faiss_status)

        # Generate recommendations
        recommendations = _generate_evaluation_recommendations(
            model_status, faiss_status
        )

        # Compile comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "type": "comprehensive_evaluation",
            },
            "system_status": {
                "model_status": model_status,
                "faiss_status": faiss_status,
                "overall_health": pipeline_health["overall_status"],
            },
            "tier_evaluation": tier_evaluation,
            "pipeline_health": pipeline_health,
            "recommendations": recommendations,
            "summary": {
                "total_models": len(model_status),
                "ready_models": sum(
                    1 for m in model_status.values() if m.get("exists", False)
                ),
                "missing_models": sum(
                    1 for m in model_status.values() if not m.get("exists", False)
                ),
                "faiss_ready": faiss_status["ready"],
                "pipeline_ready": pipeline_health["overall_status"] == "ready",
            },
        }

        logger.info("✅ Comprehensive evaluation completed successfully")
        return report

    except Exception as e:
        logger.error(f"❌ Error during comprehensive evaluation: {e}")
        return {
            "error": str(e),
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "type": "comprehensive_evaluation_error",
            },
        }


def _generate_tier_evaluation(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate tier-specific evaluation for the 3-tier architecture."""
    logger.info("🔍 Generating tier evaluation...")

    # Use centralized model configuration
    model_keys = get_all_model_keys()

    tier_evaluation = {}

    # Tier 1: Bi-Encoder Retrieval
    bi_encoder_key = "bi_encoder"
    if bi_encoder_key in model_keys:
        tier_evaluation["tier_1_retrieval"] = {
            "name": "Bi-Encoder Retrieval",
            "status": (
                "evaluated"
                if model_status.get(bi_encoder_key, {}).get("exists")
                else "missing"
            ),
            "model_path": model_status.get(bi_encoder_key, {}).get("path", "N/A"),
            "model_size_mb": model_status.get(bi_encoder_key, {}).get("size_mb", 0),
            "metrics": {
                "model_availability": model_status.get(bi_encoder_key, {}).get(
                    "exists", False
                ),
                "faiss_index_ready": faiss_status.get("exists", False),
                "retrieval_ready": model_status.get(bi_encoder_key, {}).get(
                    "exists", False
                )
                and faiss_status.get("exists", False),
            },
        }

    # Tier 2: Light Reranker
    light_reranker_key = "light_reranker"
    if light_reranker_key in model_keys:
        tier_evaluation["tier_2_light_reranking"] = {
            "name": "Light Reranking",
            "status": (
                "evaluated"
                if model_status.get(light_reranker_key, {}).get("exists")
                else "missing"
            ),
            "model_path": model_status.get(light_reranker_key, {}).get("path", "N/A"),
            "model_size_mb": model_status.get(light_reranker_key, {}).get("size_mb", 0),
            "metrics": {
                "model_availability": model_status.get(light_reranker_key, {}).get(
                    "exists", False
                ),
                "light_reranking_ready": model_status.get(light_reranker_key, {}).get(
                    "exists", False
                ),
            },
        }

    # Tier 3: Cross-Encoder
    cross_encoder_key = "cross_encoder"
    if cross_encoder_key in model_keys:
        tier_evaluation["tier_3_cross_encoder"] = {
            "name": "Cross-Encoder Ensemble",
            "status": (
                "evaluated"
                if model_status.get(cross_encoder_key, {}).get("exists")
                else "missing"
            ),
            "model_path": model_status.get(cross_encoder_key, {}).get("path", "N/A"),
            "model_size_mb": model_status.get(cross_encoder_key, {}).get("size_mb", 0),
            "metrics": {
                "model_availability": model_status.get(cross_encoder_key, {}).get(
                    "exists", False
                ),
                "cross_encoder_ready": model_status.get(cross_encoder_key, {}).get(
                    "exists", False
                ),
            },
        }

    logger.info(f"✅ Tier evaluation generated for {len(tier_evaluation)} tiers")
    return tier_evaluation


def _calculate_pipeline_health(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate overall pipeline health based on model and FAISS availability."""
    logger.info("🔍 Calculating pipeline health...")

    # Use centralized model configuration
    model_keys = get_all_model_keys()

    # Check pipeline readiness
    pipeline_ready = _check_pipeline_readiness(model_status, faiss_status)

    # Get missing components
    missing_components = _get_missing_components(model_status, faiss_status)

    # Calculate health score
    health_score = _calculate_health_score(model_status, faiss_status)

    pipeline_health = {
        "overall_status": "ready" if pipeline_ready else "not_ready",
        "health_score": health_score,
        "pipeline_ready": pipeline_ready,
        "missing_components": missing_components,
        "requirements_met": len(missing_components) == 0,
        "tier_status": {
            "tier_1": (
                "ready"
                if _is_tier_ready("tier_1", model_status, faiss_status)
                else "not_ready"
            ),
            "tier_2": (
                "ready"
                if _is_tier_ready("tier_2", model_status, faiss_status)
                else "not_ready"
            ),
            "tier_3": (
                "ready"
                if _is_tier_ready("tier_3", model_status, faiss_status)
                else "not_ready"
            ),
        },
    }

    logger.info(f"✅ Pipeline health calculated: {pipeline_health['overall_status']}")
    return pipeline_health


def _check_pipeline_readiness(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> bool:
    """Check if the entire pipeline is ready."""
    # Use centralized model configuration
    model_keys = get_all_model_keys()

    models_ready = all(
        model_status.get(model_key, {}).get("exists", False) for model_key in model_keys
    )

    faiss_ready = faiss_status.get("exists", False)

    return models_ready and faiss_ready


def _get_missing_components(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> List[str]:
    """Get list of missing components for the pipeline."""
    missing = []

    # Use centralized model configuration
    model_keys = get_all_model_keys()

    for model_key in model_keys:
        if not model_status.get(model_key, {}).get("exists", False):
            missing.append(f"Model: {get_display_name(model_key)}")

    if not faiss_status.get("exists", False):
        missing.append("FAISS Index")

    return missing


def _calculate_health_score(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> float:
    """Calculate a health score from 0 to 100."""
    # Use centralized model configuration
    model_keys = get_all_model_keys()

    total_components = len(model_keys) + 1  # +1 for FAISS
    ready_components = 0

    # Count ready models
    for model_key in model_keys:
        if model_status.get(model_key, {}).get("exists", False):
            ready_components += 1

    # Count FAISS
    if faiss_status.get("exists", False):
        ready_components += 1

    health_score = (ready_components / total_components) * 100
    return round(health_score, 1)


def _is_tier_ready(
    tier: str, model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> bool:
    """Check if a specific tier is ready."""
    if tier == "tier_1":
        # Tier 1 needs bi-encoder and FAISS
        bi_encoder_key = "bi_encoder"
        return model_status.get(bi_encoder_key, {}).get(
            "exists", False
        ) and faiss_status.get("exists", False)
    elif tier == "tier_2":
        # Tier 2 needs light reranker
        light_reranker_key = "light_reranker"
        return model_status.get(light_reranker_key, {}).get("exists", False)
    elif tier == "tier_3":
        # Tier 3 needs cross encoder
        cross_encoder_key = "cross_encoder"
        return model_status.get(cross_encoder_key, {}).get("exists", False)

    return False


def _generate_evaluation_recommendations(
    model_status: Dict[str, Any], faiss_status: Dict[str, Any]
) -> List[str]:
    """Generate actionable recommendations based on evaluation results."""
    logger.info("🔍 Generating recommendations...")

    recommendations = []

    # Use centralized model configuration
    model_keys = get_all_model_keys()

    # Check for missing models
    missing_models = []
    for model_key in model_keys:
        if not model_status.get(model_key, {}).get("exists", False):
            missing_models.append(get_display_name(model_key))

    if missing_models:
        recommendations.append(
            f"🚨 Missing models: {', '.join(missing_models)}. "
            "Run the training workflow to generate these models."
        )

    # Check FAISS index
    if not faiss_status.get("exists", False):
        recommendations.append(
            "🔍 FAISS index is missing. "
            "Ensure the bi-encoder training has completed and generated the index."
        )

    # Check model sizes
    for model_key in model_keys:
        model_info = model_status.get(model_key, {})
        if model_info.get("exists", False):
            size_mb = model_info.get("size_mb", 0)
            if size_mb < 10:  # Less than 10MB might indicate incomplete model
                recommendations.append(
                    f"⚠️ {get_display_name(model_key)} model size ({size_mb:.1f}MB) seems small. "
                    "Verify the model was trained completely."
                )

    # General recommendations
    if not recommendations:
        recommendations.append(
            "✅ All components are ready. The pipeline is fully operational."
        )
    else:
        recommendations.append(
            "💡 Run `python run_workflow.py --preset full` to train missing models and prepare the pipeline."
        )

    logger.info(f"✅ Generated {len(recommendations)} recommendations")
    return recommendations


def save_evaluation_report(report: Dict[str, Any], output_dir: Path = None) -> Path:
    """Save the evaluation report to a file."""
    if output_dir is None:
        output_dir = Path("reports")

    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_evaluation_{timestamp}.json"
    output_path = output_dir / filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Evaluation report saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Failed to save evaluation report: {e}")
        raise


def display_evaluation_summary(report: Dict[str, Any]):
    """Display a summary of the evaluation results."""
    print("\n" + "=" * 80)
    print("🔍 LAWBOB COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)

    # Summary
    summary = report.get("summary", {})
    print(f"📊 Overall Status: {report['system_status']['overall_health'].upper()}")
    print(
        f"📊 Pipeline Ready: {'✅ Yes' if summary.get('pipeline_ready') else '❌ No'}"
    )
    print(
        f"📊 Models Ready: {summary.get('ready_models', 0)}/{summary.get('total_models', 0)}"
    )
    print(f"📊 FAISS Ready: {'✅ Yes' if summary.get('faiss_ready') else '❌ No'}")

    # Tier Status
    print("\n🏗️  TIER STATUS:")
    tier_eval = report.get("tier_evaluation", {})

    if "tier_1_retrieval" in tier_eval:
        tier1 = tier_eval["tier_1_retrieval"]
        print(f"  🎯 Tier 1 (Retrieval): {tier1['status'].upper()}")
        print(
            f"    - Model Available: {'✅ Yes' if tier1['metrics']['model_availability'] else '❌ No'}"
        )
        print(
            f"    - FAISS Index Ready: {'✅ Yes' if tier1['metrics']['faiss_index_ready'] else '❌ No'}"
        )
        print(
            f"    - Retrieval Ready: {'✅ Yes' if tier1['metrics']['retrieval_ready'] else '❌ No'}"
        )

    if "tier_2_light_reranking" in tier_eval:
        tier2 = tier_eval["tier_2_light_reranking"]
        print(f"  ⚡ Tier 2 (Light Reranking): {tier2['status'].upper()}")
        print(
            f"    - Model Available: {'✅ Yes' if tier2['metrics']['model_availability'] else '❌ No'}"
        )

    if "tier_3_cross_encoder" in tier_eval:
        tier3 = tier_eval["tier_3_cross_encoder"]
        print(f"  🎯 Tier 3 (Cross-Encoder): {tier3['status'].upper()}")
        print(
            f"    - Model Available: {'✅ Yes' if tier3['metrics']['model_availability'] else '❌ No'}"
        )

    # Pipeline Health
    print(f"\n🏥 PIPELINE HEALTH: {report['pipeline_health']['health_score']}%")
    print(f"  - Overall Status: {report['pipeline_health']['overall_status'].upper()}")
    print(
        f"  - Requirements Met: {'✅ Yes' if report['pipeline_health']['requirements_met'] else '❌ No'}"
    )

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(report.get("recommendations", []), 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 80)


def main():
    """Main function to run the comprehensive evaluation."""
    logger.info("🚀 Starting LawBot comprehensive evaluation...")

    try:
        # Generate evaluation report
        report = generate_comprehensive_evaluation()

        if "error" in report:
            logger.error(f"❌ Evaluation failed: {report['error']}")
            return False

        # Save report
        output_path = save_evaluation_report(report)

        # Display summary
        display_evaluation_summary(report)

        # SỬA: Use centralized report management for fixed report
        try:
            from core.utils.report_manager import save_report, setup_report_management
            
            # Setup report management
            setup_report_management()
            
            # Save with fixed name using centralized management
            save_report(
                report_type="comprehensive_evaluation",
                data=report,
                timestamp="latest",  # Use latest timestamp
                keep_latest_only=False
            )
            
            # Also save as fixed name for backward compatibility
            fixed_path = Path("reports/comprehensive_evaluation.json")
            with open(fixed_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Fixed report saved to: {fixed_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save fixed report: {e}")

        logger.info("✅ Comprehensive evaluation completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Unexpected error during evaluation: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
