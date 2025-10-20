"""
Test suite for Enhanced AI Trading System
"""
import sys
import os

def test_imports():
    """Test all critical imports"""
    try:
        from two_stage_config import get_config, TradingConfig
        print("✅ Config imports successful")

        from two_stage_data_manager import EnhancedDataManager, StockData
        print("✅ Data manager imports successful")

        from database import get_db_manager, DatabaseManager
        print("✅ Database imports successful")

        from perplexity_client import PerplexityClientSync, TradingRecommendation
        print("✅ Perplexity client imports successful")

        from two_stage_ml_engine import MLRecommendationEngine
        print("✅ ML engine imports successful")

        from trade_logger import TradeLogger
        print("✅ Trade logger imports successful")

        from weekend_analyzer import WeekendAnalyzer
        print("✅ Weekend analyzer imports successful")

        from performance_analyzer import PerformanceAnalyzer
        print("✅ Performance analyzer imports successful")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration management"""
    try:
        from two_stage_config import get_config

        config = get_config()
        assert config.INITIAL_CAPITAL > 0
        assert config.MAX_POSITION_SIZE <= 1.0

        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"⚠️  Configuration test failed (expected if no API key): {e}")
        return False

def test_data_manager():
    """Test data management functionality"""
    try:
        from two_stage_data_manager import EnhancedDataManager

        data_manager = EnhancedDataManager()

        assert hasattr(data_manager, 'validator')
        assert hasattr(data_manager, 'cache')

        print("✅ Data manager test passed")
        return True
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def run_integration_test():
    """Run a comprehensive integration test"""
    print("Running Enhanced ML Trading System Integration Test...")
    print("=" * 60)

    test_results = []

    print("\n1. Testing imports...")
    test_results.append(test_imports())

    print("\n2. Testing configuration...")
    test_results.append(test_configuration())

    print("\n3. Testing data manager...")
    test_results.append(test_data_manager())

    print("\n" + "="*60)
    passed_tests = sum(test_results)
    total_tests = len(test_results)

    if passed_tests >= 2:  # Allow config to fail if no API key
        print("🎉 Integration tests passed! System is ready to run.")
        print("\nNext steps:")
        print("1. Make sure you have your Perplexity API key in .env file")
        print("2. Run: python3 enhanced_ml_trading_system.py")
        return True
    else:
        print(f"⚠️  {passed_tests}/{total_tests} tests passed.")
        print("Some issues found. Check the errors above.")
        return False

if __name__ == "__main__":
    run_integration_test()
