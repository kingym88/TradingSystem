"""
Two-Stage Enhanced Configuration for AI Trading System
UPDATED: Optimized for 4000 → 50 → 10 intelligent stock selection
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

class TwoStageEnhancedConfig(BaseSettings):
    """Two-stage configuration class optimized for large-scale stock analysis"""
    
    # API Configuration
    PERPLEXITY_API_KEY: str
    PERPLEXITY_BASE_URL: str = "https://api.perplexity.ai"
    
    # Enhanced API keys for alternative data
    ALPHA_VANTAGE_KEY: Optional[str] = None
    FRED_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    FINNHUB_API_KEY: Optional[str] = None
    EODHD_API_KEY: Optional[str] = None
    
    # ==========================================
    # TWO-STAGE FILTERING CONFIGURATION
    # ==========================================
    
    # Stage 1: Fast Screening (4000+ → 50)
    STAGE1_TARGET_COUNT: int = 200
    STAGE1_MAX_MARKET_CAP: int = 50_000_000_000  # $50B max for screening
    STAGE1_MIN_VOLUME: int = 1_000_000  # 1M minimum volume
    STAGE1_MIN_PRICE: float = 1.00  # $1 minimum price
    STAGE1_MAX_PRICE: float = 100.00  # $1000 maximum price
    STAGE1_BATCH_SIZE: int = 1000  # Process in batches
    STAGE1_PARALLEL_WORKERS: int = 5  # Parallel processing
    
    # Stage 2: Detailed Analysis (50 → 10)
    STAGE2_TARGET_COUNT: int = 50
    STAGE2_BATCH_SIZE: int = 50  # Smaller batches for detailed analysis
    STAGE2_ENABLE_SENTIMENT: bool = True
    STAGE2_ENABLE_ADVANCED_ML: bool = True
    STAGE2_ENABLE_BAYESIAN: bool = True
    
    # Performance Optimization
    ENABLE_CACHING: bool = True
    CACHE_DURATION_HOURS: int = 1
    MAX_CONCURRENT_REQUESTS: int = 50
    REQUEST_DELAY_SECONDS: float = 0.1
    API_TIMEOUT_SECONDS: int = 10
    
    # ==========================================
    # TRADING PARAMETERS - Enhanced
    # ==========================================
    
    INITIAL_CAPITAL: float = 5000.0
    MAX_POSITION_SIZE: float = 0.25  # 25% max per position
    STOP_LOSS_PERCENTAGE: float = 0.15  # 15% stop loss
    TAKE_PROFIT_PERCENTAGE: float = 0.25  # 25% take profit
    
    # Kelly Criterion Parameters - Enhanced for Two-Stage
    KELLY_FRACTION: float = 0.5  # Use 50% of Kelly recommendation
    CONFIDENCE_THRESHOLD: float = 0.89  # Kelly threshold
    MIN_CONFIDENCE_FOR_TRADE: float = 0.65  # Minimum confidence required
    TWO_STAGE_CONFIDENCE_BOOST: float = 0.05  # Bonus for two-stage selections
    
    # Risk Management - Enhanced
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    MAX_POSITIONS: int = 50
    MIN_VOLUME: int = 1_000_000  # Minimum daily volume (used in stage 2)
    MAX_VOLATILITY: float = 0.08  # Maximum volatility threshold
    MIN_VOLATILITY: float = 0.01  # Minimum volatility for liquidity
    
    # ==========================================
    # ML CONFIGURATION - Two-Stage Optimized
    # ==========================================
    
    MAX_DAILY_RECOMMENDATIONS: int = 20
    ML_RETRAIN_FREQUENCY: int = 7  # Retrain ML model every 7 days
    FEATURE_WINDOW: int = 30  # Look back 30 days for features
    MIN_TRAINING_SAMPLES: int = 20  # Reduced for faster learning
    ENABLE_MODEL_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = "sigmoid"  # or "isotonic"
    ENSEMBLE_SIZE: int = 3  # Reduced for performance
    
    # ==========================================
    # TECHNICAL ANALYSIS PARAMETERS
    # ==========================================
    
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    ADX_PERIOD: int = 14
    ADX_THRESHOLD: float = 25.0  # Strong trend threshold
    
    # ==========================================
    # SENTIMENT ANALYSIS PARAMETERS
    # ==========================================
    
    NEWS_LOOKBACK_DAYS: int = 3  # Reduced for performance
    SENTIMENT_THRESHOLD: float = 0.1  # Minimum sentiment score
    SENTIMENT_WEIGHT: float = 0.10  # Reduced weight for performance
    ENABLE_NEWS_SENTIMENT: bool = True
    MAX_NEWS_ARTICLES: int = 10  # Reduced for performance
    
    # ==========================================
    # BAYESIAN PARAMETERS
    # ==========================================
    
    ENABLE_BAYESIAN_UPDATING: bool = True
    BAYESIAN_PRIOR_ALPHA: float = 5.0  # Beta distribution prior
    BAYESIAN_PRIOR_BETA: float = 5.0
    BAYESIAN_UPDATE_FREQUENCY: int = 3  # Update every 3 trades (faster learning)
    BAYESIAN_MIN_TRADES: int = 5  # Minimum trades for Bayesian update
    
    # ==========================================
    # BACKTESTING PARAMETERS
    # ==========================================
    
    BACKTEST_PERIOD: str = "1y"  # Reduced for performance
    BACKTEST_TRAIN_RATIO: float = 0.8  # 80% for training
    MIN_BACKTEST_TRADES: int = 30  # Minimum trades for reliable backtest
    CONFIDENCE_INTERVAL: float = 0.95  # For confidence bounds
    ENABLE_WALK_FORWARD: bool = False  # Disabled for performance
    
    # ==========================================
    # EXPERT RULES AND FILTERS
    # ==========================================
    
    ENABLE_VOLUME_FILTER: bool = True
    VOLUME_SURGE_THRESHOLD: float = 1.5  # 1.5x average volume (reduced)
    ENABLE_BREAKOUT_FILTER: bool = True
    ENABLE_MARKET_REGIME_FILTER: bool = True
    SECTOR_CONCENTRATION_LIMIT: float = 0.4  # Max 40% in any sector
    
    # ==========================================
    # REAL-TIME MONITORING
    # ==========================================
    
    ENABLE_PERFORMANCE_TRACKING: bool = True
    CALIBRATION_WINDOW: int = 15  # Rolling window for calibration (reduced)
    CONFIDENCE_ADJUSTMENT_FACTOR: float = 0.05  # Max adjustment per period
    MODEL_DRIFT_THRESHOLD: float = 0.05  # When to retrain model
    
    # ==========================================
    # DATABASE & LOGGING
    # ==========================================
    
    DATABASE_URL: str = "sqlite:///two_stage_trading_system.db"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/two_stage_trading_system.log"
    ENABLE_DEBUG_LOGGING: bool = False
    
    # ==========================================
    # MARKET HOURS & TIMING
    # ==========================================
    
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    
    # Research Settings - Two-Stage Optimized
    DEEP_RESEARCH_DAYS: List[int] = [5, 6]  # Saturday = 5, Sunday = 6
    MAX_RESEARCH_STOCKS: int = 50  # Aligned with Stage 1 target
    
    # Weekend Analysis - Streamlined
    WEEKEND_ANALYSIS_ENABLED: bool = True
    WEEKEND_PORTFOLIO_REVIEW: bool = True
    WEEKEND_MARKET_ANALYSIS: bool = True
    WEEKEND_MODEL_RETRAIN: bool = False  # Disabled for performance
    
    # ==========================================
    # PERFORMANCE TARGETS
    # ==========================================
    
    # Expected execution times (in seconds)
    TARGET_STAGE1_TIME: int = 120  # 2 minutes for stage 1
    TARGET_STAGE2_TIME: int = 300  # 5 minutes for stage 2
    TARGET_TOTAL_TIME: int = 480   # 8 minutes total
    MAX_ACCEPTABLE_TIME: int = 900  # 15 minutes maximum
    
    # Performance alerts
    ENABLE_PERFORMANCE_ALERTS: bool = True
    SLOW_EXECUTION_THRESHOLD: float = 1.5  # Alert if 1.5x target time
    
    # ==========================================
    # FILE PATHS
    # ==========================================
    
    DATA_DIR: Path = Path("data")
    LOGS_DIR: Path = Path("logs")
    REPORTS_DIR: Path = Path("reports")
    MODELS_DIR: Path = Path("models")
    CACHE_DIR: Path = Path("cache")
    
    # ==========================================
    # VALIDATION RULES
    # ==========================================
    
    @field_validator("PERPLEXITY_API_KEY")
    def validate_api_key(cls, v):
        if not v or v == "your_perplexity_api_key_here":
            raise ValueError("PERPLEXITY_API_KEY is required - please set it in your .env file")
        return v
    
    @field_validator("INITIAL_CAPITAL")
    def validate_capital(cls, v):
        if v <= 0:
            raise ValueError("INITIAL_CAPITAL must be positive")
        return v
    
    @field_validator("STAGE1_TARGET_COUNT")
    def validate_stage1_target(cls, v):
        if v < 10 or v > 200:
            raise ValueError("STAGE1_TARGET_COUNT must be between 10 and 200")
        return v
    
    @field_validator("STAGE2_TARGET_COUNT")
    def validate_stage2_target(cls, v):
        if v < 5 or v > 50:
            raise ValueError("STAGE2_TARGET_COUNT must be between 5 and 50")
        return v
    
    @field_validator("KELLY_FRACTION")
    def validate_kelly_fraction(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("KELLY_FRACTION must be between 0 and 1")
        return v
    
    @field_validator("CONFIDENCE_THRESHOLD")
    def validate_confidence_threshold(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.REPORTS_DIR, 
                         self.MODELS_DIR, self.CACHE_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global config instance
config = None

def get_config() -> TwoStageEnhancedConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        try:
            config = TwoStageEnhancedConfig()
        except ValueError as e:
            print(f"Configuration Error: {e}")
            print("Please check your .env file and ensure all required API keys are set")
            raise
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    if config is None:
        config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")

def get_performance_targets() -> dict:
    """Get performance targets for monitoring"""
    cfg = get_config()
    return {
        'stage1_time': cfg.TARGET_STAGE1_TIME,
        'stage2_time': cfg.TARGET_STAGE2_TIME,
        'total_time': cfg.TARGET_TOTAL_TIME,
        'max_time': cfg.MAX_ACCEPTABLE_TIME
    }

def get_two_stage_config() -> dict:
    """Get two-stage specific configuration"""
    cfg = get_config()
    return {
        'stage1_target': cfg.STAGE1_TARGET_COUNT,
        'stage2_target': cfg.STAGE2_TARGET_COUNT,
        'stage1_filters': {
            'max_market_cap': cfg.STAGE1_MAX_MARKET_CAP,
            'min_volume': cfg.STAGE1_MIN_VOLUME,
            'min_price': cfg.STAGE1_MIN_PRICE,
            'max_price': cfg.STAGE1_MAX_PRICE
        },
        'stage2_features': {
            'sentiment_enabled': cfg.STAGE2_ENABLE_SENTIMENT,
            'advanced_ml_enabled': cfg.STAGE2_ENABLE_ADVANCED_ML,
            'bayesian_enabled': cfg.STAGE2_ENABLE_BAYESIAN
        }
    }

def get_api_keys() -> dict:
    """Get all available API keys"""
    cfg = get_config()
    return {
        'perplexity': cfg.PERPLEXITY_API_KEY,
        'alpha_vantage': cfg.ALPHA_VANTAGE_KEY,
        'news_api': cfg.NEWS_API_KEY,
        'finnhub': cfg.FINNHUB_API_KEY,
        'eodhd': cfg.EODHD_API_KEY,
        'fred': cfg.FRED_API_KEY
    }

def is_performance_mode() -> bool:
    """Check if system should run in performance mode"""
    cfg = get_config()
    # Performance mode if we have large targets or limited APIs
    return cfg.STAGE1_TARGET_COUNT >= 100 or not cfg.ALPHA_VANTAGE_KEY

def get_optimized_settings() -> dict:
    """Get optimized settings based on available resources"""
    cfg = get_config()
    
    # Adjust settings based on available APIs and resources
    if not cfg.ALPHA_VANTAGE_KEY:
        # No premium APIs - optimize for speed
        return {
            'sentiment_enabled': False,
            'news_articles': 0,
            'batch_size': 20,
            'concurrent_requests': 3
        }
    else:
        # Premium APIs available - enable features
        return {
            'sentiment_enabled': True,
            'news_articles': cfg.MAX_NEWS_ARTICLES,
            'batch_size': cfg.STAGE1_BATCH_SIZE,
            'concurrent_requests': cfg.MAX_CONCURRENT_REQUESTS
        }