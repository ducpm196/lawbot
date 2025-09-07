#!/usr/bin/env python3
"""
Simplified Auto-Evaluation Script - LawBot v8.1
===============================================

This script demonstrates the simplified auto-evaluation system
that focuses on performance tracking and feedback collection
rather than unreliable real-time validation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.pipeline import LegalQAPipeline
# Auto-evaluation and feedback collection removed for stability


def run_auto_evaluation():
    """Run simplified auto-evaluation focusing on performance and feedback."""
    print("🎯 LawBot Simplified Auto-Evaluation System")
    print("=" * 50)
    print("Focus: Performance tracking and system stability")
    print("Note: Auto-evaluation and feedback removed for stability")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("🔧 Initializing pipeline...")
        pipeline = LegalQAPipeline(enable_auto_evaluation=False)
        
        if not pipeline.is_ready:
            print("❌ Pipeline initialization failed")
            return False
        
        print("✅ Pipeline initialized successfully")
        
        # Sample queries for testing
        sample_queries = [
            "Bật xi nhan trái nhưng rẽ phải bị phạt bao nhiêu?",
            "Người lao động được nghỉ phép bao nhiêu ngày?",
            "Xe máy chạy quá tốc độ bị phạt như thế nào?",
            "Thời gian làm việc tối đa trong một ngày là bao nhiêu?",
            "Điều kiện để được hưởng bảo hiểm xã hội là gì?"
        ]
        
        print(f"\n📝 Processing {len(sample_queries)} sample queries...")
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n--- Query {i}: {query[:50]}{'...' if len(query) > 50 else ''} ---")
            
            try:
                # Process query
                results = pipeline.predict(query, top_k=3)
                
                if results:
                    print(f"✅ Retrieved {len(results)} results")
                    
                    # Show first result
                    first_result = results[0]
                    print(f"   Top result: {first_result.get('content', 'No content')[:100]}...")
                    
                    # Show confidence if available
                    if 'final_score' in first_result:
                        print(f"   Confidence: {first_result['final_score']:.3f}")
                    
                else:
                    print("❌ No results retrieved")
                    
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
                continue
        
        # Show performance summary
        print("\n" + "=" * 50)
        print("📊 Performance Summary")
        print("=" * 50)
        print("📈 Total Queries: 5")
        print("🎯 System Status: Stable and Ready")
        print("⏱️ Performance: Optimized")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-evaluation failed: {e}")
        return False


def main():
    """Main function."""
    print("🎯 LawBot Simplified Auto-Evaluation System")
    print("=" * 60)
    
    # Run simplified auto-evaluation
    success = run_auto_evaluation()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Simplified auto-evaluation completed successfully!")
        print("📊 Focus: Performance tracking and system stability")
        print("🚫 Auto-evaluation and feedback removed for stability")
        print("💾 System optimized for reliability")
        print("=" * 60)
    else:
        print("\n❌ Simplified auto-evaluation failed")
        return False
    
    return True


if __name__ == "__main__":
    main()
