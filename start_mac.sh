#!/bin/bash
echo "🚀 Enhanced ML AI Trading System - Mac Startup"
echo "============================================="

# Check if Python 3.12 is available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    echo "✅ Using Python 3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    echo "✅ Using Python 3"
else
    echo "❌ Python 3 not found. Please install Python 3.11"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "trading_env" ]; then
    echo "Creating Python 3.12 virtual environment..."
    $PYTHON_CMD -m venv trading_env
    source trading_env/bin/activate
    python -m pip install --upgrade pip
    echo "Installing ML dependencies..."
    pip install -r enhanced_requirements.txt
else
    echo "Activating virtual environment..."
    source trading_env/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Configuration file not found!"
    echo "Please copy enhanced_env_template to .env and add your Perplexity API key"
    echo "Example: cp enhanced_env_template .env"
    echo "Then edit .env and add: PERPLEXITY_API_KEY=your_key_here"
    exit 1
fi

# Create required directories
mkdir -p logs data models reports

# Run the enhanced ML system
echo "🚀 Starting Enhanced ML AI Trading System with Profitable Stock Analysis..."
echo "Features:"
echo "  🧠 AI-Powered Profitability Analysis"
echo "  📊 Dynamic ETF Holdings Screening"  
echo "  📝 Interactive Trade Logging"
echo "  📅 Weekend Deep Analysis"
echo "  ⚠️  Advanced Risk Management"
echo ""
$PYTHON_CMD enhanced_ml_trading_system.py

echo ""
echo "System closed. Thanks for using Enhanced ML AI Trading System! 🚀"
