#!/usr/bin/env python3
"""
Simple Two-Stage Analysis Script
Works with your existing AI trading bot system
"""

import sys
import time
from typing import List

def run_two_stage_analysis():
    """Simple two-stage analysis using existing system"""
    try:
        print("🎯 Simple Two-Stage Stock Analysis")
        print("="*50)
        
        # Import your existing system
        from two_stage_data_manager import EnhancedDataManager
        from two_stage_ml_engine import MLRecommendationEngine
        
        data_manager = EnhancedDataManager()
        ml_engine = MLRecommendationEngine()
        
        start_time = time.time()
        
        # STAGE 1: Fast screening (limit processing)
        print("🚀 Stage 1: Fast screening...")
        candidates = data_manager.screen_micro_caps()
        
        if not candidates:
            print("❌ No candidates found")
            return
        
        # Limit to top 50 instead of processing all 4000+
        stage1_candidates = candidates[:50]
        stage1_time = time.time() - start_time
        
        print(f"✅ Stage 1: {len(candidates)} → {len(stage1_candidates)} candidates ({stage1_time:.1f}s)")
        
        # STAGE 2: Generate final recommendations
        print("🔬 Stage 2: ML recommendations...")
        recommendations = ml_engine.generate_recommendations(10)
        
        total_time = time.time() - start_time
        
        # Results
        print(f"\n✅ Analysis complete in {total_time:.1f} seconds!")
        print(f"📊 Flow: {len(candidates)} → {len(stage1_candidates)} → {len(recommendations) if recommendations else 0}")
        
        if recommendations:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                symbol = rec.get('symbol', 'N/A')
                confidence = rec.get('confidence', 0)
                print(f"  {i}. {symbol}: {confidence:.1%} confidence")
        
    except ImportError as e:
        print(f"❌ Missing components: {e}")
        print("Please ensure your original system files are present")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_two_stage_analysis()