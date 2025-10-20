#!/bin/bash

# Two-Stage AI Trading System Setup Script
# Handles installation and configuration for optimal performance

echo "🎯 Two-Stage AI Trading System Setup"
echo "===================================="
echo "Optimized for 4000 → 50 → 10 intelligent stock filtering"
echo ""

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Python version: $python_version"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment (recommended)
read -p "🔧 Create virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv two_stage_trading_env
    source two_stage_trading_env/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies based on user preference
echo ""
echo "📦 Installation Options:"
echo "1. Minimal (Core functionality only) - ~30 seconds"
echo "2. Recommended (Enhanced features) - ~2 minutes"  
echo "3. Maximum (All features) - ~5 minutes"
echo ""

read -p "Select installation level (1-3): " install_level

case $install_level in
    1)
        echo "📦 Installing minimal dependencies..."
        pip install pandas numpy yfinance requests python-dotenv loguru sqlalchemy pydantic pydantic-settings aiohttp
        echo "✅ Minimal installation complete"
        ;;
    2)
        echo "📦 Installing recommended dependencies..."
        pip install pandas numpy yfinance requests python-dotenv loguru sqlalchemy pydantic pydantic-settings
        pip install pandas-ta scikit-learn scipy
        echo "✅ Recommended installation complete"
        ;;
    3)
        echo "📦 Installing maximum dependencies..."
        pip install pandas numpy yfinance requests python-dotenv loguru sqlalchemy pydantic pydantic-settings
        pip install pandas-ta scikit-learn scipy xgboost lightgbm
        pip install vaderSentiment transformers torch
        pip install backtrader vectorbt
        echo "✅ Maximum installation complete"
        ;;
    *)
        echo "❌ Invalid selection. Installing recommended dependencies..."
        pip install pandas numpy yfinance requests python-dotenv loguru sqlalchemy pydantic pydantic-settings
        pip install pandas-ta scikit-learn scipy
        ;;
esac

# Setup configuration
echo ""
echo "⚙️ Configuration Setup"
echo "===================="

if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    read -p "Overwrite existing .env file? (y/n): " overwrite_env
    if [[ $overwrite_env =~ ^[Yy]$ ]]; then
        cp two_stage_env_template.txt .env
        echo "✅ Configuration template copied to .env"
    fi
else
    cp two_stage_env_template.txt .env
    echo "✅ Configuration template copied to .env"
fi

# API Key setup
echo ""
echo "🔑 API Key Configuration"
echo "======================="
echo "Required: Perplexity AI API Key"
echo "Optional: Alpha Vantage, NewsAPI, Finnhub (for enhanced features)"
echo ""

read -p "Configure API keys now? (y/n): " configure_keys
if [[ $configure_keys =~ ^[Yy]$ ]]; then
    read -p "Enter Perplexity API Key: " perplexity_key
    if [ ! -z "$perplexity_key" ]; then
        sed -i.bak "s/your_perplexity_api_key_here/$perplexity_key/" .env
        echo "✅ Perplexity API key configured"
    fi
    
    read -p "Enter Alpha Vantage API Key (optional): " alpha_vantage_key
    if [ ! -z "$alpha_vantage_key" ]; then
        sed -i.bak "s/your_alpha_vantage_key_here/$alpha_vantage_key/" .env
        echo "✅ Alpha Vantage API key configured"
    fi
fi

# Performance mode selection
echo ""
echo "⚡ Performance Mode Selection"
echo "==========================="
echo "1. Fast Mode (3-4 minutes, basic features)"
echo "2. Balanced Mode (5-8 minutes, recommended)"
echo "3. Comprehensive Mode (8-12 minutes, all features)"
echo ""

read -p "Select performance mode (1-3, default: 2): " perf_mode
perf_mode=${perf_mode:-2}

case $perf_mode in
    1)
        echo "⚡ Configuring Fast Mode..."
        cat >> .env << EOF

# Fast Mode Configuration
STAGE1_TARGET_COUNT=30
STAGE2_TARGET_COUNT=5
STAGE2_ENABLE_SENTIMENT=false
STAGE2_ENABLE_BAYESIAN=false
MAX_NEWS_ARTICLES=5
EOF
        echo "✅ Fast Mode configured"
        ;;
    2) 
        echo "⚡ Using Balanced Mode (default configuration)..."
        echo "✅ Balanced Mode configured"
        ;;
    3)
        echo "⚡ Configuring Comprehensive Mode..."
        cat >> .env << EOF

# Comprehensive Mode Configuration  
STAGE1_TARGET_COUNT=100
STAGE2_TARGET_COUNT=15
STAGE2_ENABLE_SENTIMENT=true
STAGE2_ENABLE_ADVANCED_ML=true
MAX_NEWS_ARTICLES=20
EOF
        echo "✅ Comprehensive Mode configured"
        ;;
esac

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data logs reports models cache
echo "✅ Directories created"

# Final setup verification
echo ""
echo "🔧 Setup Verification"
echo "==================="

# Test imports
echo "Testing core imports..."
python3 -c "
try:
    import pandas, numpy, yfinance, requests, loguru
    print('✅ Core dependencies working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test enhanced imports
echo "Testing enhanced imports..."
python3 -c "
try:
    import pandas_ta
    print('✅ Technical analysis available')
except ImportError:
    print('⚠️  pandas-ta not available (basic indicators will be used)')

try:
    import sklearn
    print('✅ Machine learning available')
except ImportError:
    print('⚠️  scikit-learn not available (basic ML will be used)')
"

echo ""
echo "🎯 Setup Complete!"
echo "================="
echo ""
echo "📊 System Configuration:"
case $install_level in
    1) echo "   Installation: Minimal (core functionality)" ;;
    2) echo "   Installation: Recommended (enhanced features)" ;;
    3) echo "   Installation: Maximum (all features)" ;;
esac

case $perf_mode in
    1) echo "   Performance: Fast Mode (3-4 minutes)" ;;
    2) echo "   Performance: Balanced Mode (5-8 minutes)" ;;
    3) echo "   Performance: Comprehensive Mode (8-12 minutes)" ;;
esac

echo ""
echo "🚀 Ready to Start!"
echo "=================="
echo ""
echo "To run the two-stage trading system:"
echo "   python two_stage_trading_system.py"
echo ""
echo "Expected analysis time: 5-8 minutes for 4000+ stocks"
echo "Expected output: 10 high-quality recommendations"
echo ""

# Offer to run the system
read -p "🚀 Run the system now? (y/n): " run_now
if [[ $run_now =~ ^[Yy]$ ]]; then
    echo ""
    echo "🎯 Starting Two-Stage AI Trading System..."
    echo "Expected runtime: 5-8 minutes"
    echo "Analysis flow: 4000+ stocks → 50 candidates → 10 recommendations"
    echo ""
    python3 two_stage_trading_system.py
else
    echo ""
    echo "👍 Setup complete! Run the system anytime with:"
    echo "   python two_stage_trading_system.py"
    echo ""
    echo "📖 For detailed information, see: TWO_STAGE_README.md"
fi

echo ""
echo "🎯 Two-Stage AI Trading System ready!"
echo "Process 4000+ stocks in under 8 minutes with intelligent filtering!"