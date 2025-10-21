"""
Two-Stage Enhanced ML Trading System
UPDATED: Enhanced display for portfolio stocks vs new stocks with sell prices
"""
import sys
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import time

# Import configuration
try:
    from two_stage_config import get_config
    config = get_config()
except ImportError:
    logger.error("❌ Cannot import enhanced_config. Using fallback.")
    class FallbackConfig:
        INITIAL_CAPITAL = 5000.0
        KELLY_FRACTION = 0.5
        CONFIDENCE_THRESHOLD = 0.89
        MIN_CONFIDENCE_FOR_TRADE = 0.65
        MAX_POSITION_SIZE = 0.25
        LOG_LEVEL = "INFO"
        LOG_FILE = "logs/two_stage_trading_system.log"
    config = FallbackConfig()

# Configure enhanced logging
logger.remove()
logger.add(sys.stderr, level=config.LOG_LEVEL,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
logger.add(config.LOG_FILE, rotation="10 MB", retention="2 months", level=config.LOG_LEVEL,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")

# Import system components
from database import get_db_manager
from two_stage_data_manager import get_two_stage_data_manager
from two_stage_ml_engine import get_two_stage_ml_engine
from portfolio_manager import get_portfolio_manager


class TwoStageMLTradingSystem:
    """
    Main trading system with two-stage stock analysis (4000 → 50 → 10)
    UPDATED: Enhanced portfolio tracking and recommendation display
    """
    
    def __init__(self):
        self.config = config
        self.portfolio_manager = get_portfolio_manager()
        self.data_manager = get_two_stage_data_manager()
        self.ml_engine = get_two_stage_ml_engine()
        self.db_manager = get_db_manager()
        
        logger.info("🚀 Two-Stage ML Trading System initialized (4000 → 50 → 10)")
    
    async def run_two_stage_daily_update(self) -> None:
        """Run enhanced daily update with two-stage analysis"""
        try:
            start_time = time.time()
            logger.info("🎯 Starting Two-Stage Daily Update (4000 → 50 → 10)...")
            
            # Get current portfolio with live prices
            logger.info("📊 Fetching portfolio with live prices from Yahoo Finance...")
            portfolio_data = self.portfolio_manager.get_portfolio_summary()
            
            # Display enhanced portfolio summary
            self._display_enhanced_portfolio_summary(portfolio_data)
            
            # Learn from recent trades
            logger.info("🧠 Learning from trade history...")
            self.ml_engine.learn_from_trades()
            
            # Generate two-stage recommendations (split into portfolio and new stocks)
            logger.info("🎯 Running Two-Stage Analysis...")
            recommendations = await self.ml_engine.generate_recommendations()
            
            if not recommendations:
                logger.warning("❌ No recommendations generated from two-stage analysis")
                return
            
            # Separate recommendations by type
            portfolio_recs = [r for r in recommendations if r.get('is_portfolio_stock', False)]
            new_stock_recs = [r for r in recommendations if not r.get('is_portfolio_stock', False)]
            
            # Display recommendations
            self._display_two_stage_recommendations(recommendations, portfolio_data, portfolio_recs, new_stock_recs)
            
            # Simulate trade execution for high-confidence recommendations
            executed_trades = 0
            for rec in recommendations:
                try:
                    if rec['confidence'] >= self.config.CONFIDENCE_THRESHOLD and rec.get('action') == 'BUY':
                        # Simulate trade execution for BUY recommendations only
                        success, message = self._simulate_trade_execution(rec, portfolio_data)
                        if success:
                            executed_trades += 1
                            logger.info(f"✅ Simulated trade: {message}")
                        else:
                            logger.warning(f"❌ Trade simulation failed: {message}")
                    elif rec['confidence'] >= self.config.MIN_CONFIDENCE_FOR_TRADE:
                        logger.info(f"⚠️ Moderate confidence: {rec['symbol']} "
                                  f"({rec['confidence']:.1%}) - manual review recommended")
                except Exception as e:
                    logger.error(f"Error processing {rec['symbol']}: {e}")
            
            # Performance summary
            total_time = time.time() - start_time
            logger.info(f"✅ Two-Stage Daily Update Complete!")
            logger.info(f"⏱️ Total time: {total_time:.1f} seconds")
            logger.info(f"📊 Portfolio stocks: {len(portfolio_recs)}, New opportunities: {len(new_stock_recs)}")
            logger.info(f"🎯 High-confidence trades: {executed_trades}")
        
        except Exception as e:
            logger.error(f"Error in two-stage daily update: {e}")
    
    def _display_enhanced_portfolio_summary(self, portfolio_data: Dict) -> None:
        """
        UPDATED: Display enhanced portfolio summary with live prices and P/L
        """
        try:
            print("\n" + "="*100)
            print("💼 CURRENT PORTFOLIO STATUS (Live Prices from Yahoo Finance)")
            print("="*100)
            
            print(f"\n📊 PORTFOLIO OVERVIEW:")
            print(f"  Total Portfolio Value: ${portfolio_data['total_value']:,.2f}")
            print(f"  Cash Available: ${portfolio_data['cash']:,.2f}")
            print(f"  Total Invested (Cost Basis): ${portfolio_data['invested_amount']:,.2f}")
            print(f"  Current Market Value: ${portfolio_data['current_market_value']:,.2f}")
            print(f"  Total Return: ${portfolio_data['total_return_dollars']:+,.2f} ({portfolio_data['total_return']:+.2f}%)")
            print(f"  Active Positions: {portfolio_data['num_positions']}")
            
            # Display individual holdings with live prices
            if portfolio_data['holdings_details']:
                print(f"\n📋 CURRENT HOLDINGS (Live Market Data):")
                print("-" * 100)
                print(f"{'Symbol':<8} {'Shares':<8} {'Avg Buy':<12} {'Current':<12} {'Value':<12} {'P/L $':<12} {'P/L %':<10}")
                print("-" * 100)
                
                for symbol, details in portfolio_data['holdings_details'].items():
                    print(f"{symbol:<8} "
                          f"{details['quantity']:<8,} "
                          f"${details['avg_buy_price']:<11.2f} "
                          f"${details['current_price']:<11.2f} "
                          f"${details['current_value']:<11,.2f} "
                          f"${details['unrealized_pnl']:<+11.2f} "
                          f"{details['unrealized_pnl_percent']:<+9.1f}%")
                
                print("-" * 100)
            else:
                print("\n📋 No active positions")
            
            print("="*100)
        
        except Exception as e:
            logger.error(f"Error displaying portfolio summary: {e}")
    
    def _simulate_trade_execution(self, recommendation: Dict, portfolio_data: Dict) -> Tuple[bool, str]:
        """Simulate trade execution (for demonstration)"""
        try:
            symbol = recommendation['symbol']
            price = recommendation['current_price']
            confidence = recommendation['confidence']
            expected_return = recommendation.get('expected_return', 0.12)
            
            # Calculate position size
            position_size = self.portfolio_manager.get_enhanced_kelly_position_size(
                price, expected_return, confidence, portfolio_data['total_value']
            )
            
            if position_size <= 0:
                return False, f"Kelly sizing resulted in zero position"
            
            quantity = int(position_size // price)
            if quantity == 0:
                return False, f"Position too small for single share"
            
            total_cost = quantity * price
            
            if total_cost > portfolio_data['cash']:
                return False, f"Insufficient funds: ${total_cost:.2f} > ${portfolio_data['cash']:.2f}"
            
            # Save simulated trade
            trade_data = {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'total_amount': total_cost,
                'reasoning': f"Two-stage analysis: {recommendation.get('reasoning', 'High confidence')}",
                'confidence': confidence,
                'timestamp': datetime.now()
            }
            
            self.db_manager.save_trade(trade_data)
            
            return True, f"BUY {quantity} {symbol} @ ${price:.2f} = ${total_cost:.2f} (Confidence: {confidence:.1%})"
        
        except Exception as e:
            return False, f"Trade simulation error: {str(e)}"
    
    def _display_two_stage_recommendations(self, recommendations: List[Dict], portfolio_data: Dict,
                                          portfolio_recs: List[Dict], new_stock_recs: List[Dict]) -> None:
        """
        UPDATED: Display two-stage analysis results with portfolio stocks separated
        """
        try:
            print("\n" + "="*100)
            print("🎯 TWO-STAGE ML TRADING SYSTEM - INTELLIGENT RECOMMENDATIONS")
            print("="*100)
            print("📊 ANALYSIS FLOW: 4000+ stocks → Top 50 candidates → Final recommendations")
            print(f"💰 Portfolio Value: ${portfolio_data['total_value']:.2f} | Return: {portfolio_data['total_return']:+.2f}%")
            
            if not recommendations:
                print("\n❌ No recommendations generated")
                return
            
            # SECTION 1: Portfolio Stock Recommendations (BUY/SELL existing positions)
            if portfolio_recs:
                print("\n" + "="*100)
                print("📊 SECTION 1: PORTFOLIO STOCK RECOMMENDATIONS (Existing Positions)")
                print("="*100)
                print(f"You currently own {len(portfolio_recs)} of the recommended stocks. Here's what to do:")
                print()
                
                for i, rec in enumerate(portfolio_recs, 1):
                    self._display_portfolio_stock_recommendation(i, rec)
            
            # SECTION 2: New Stock Recommendations (BUY new positions)
            if new_stock_recs:
                print("\n" + "="*100)
                print("🆕 SECTION 2: NEW STOCK RECOMMENDATIONS (New Opportunities)")
                print("="*100)
                print(f"Fresh opportunities identified by two-stage analysis:")
                print()
                
                # Categorize by confidence
                high_confidence = [r for r in new_stock_recs if r['confidence'] >= self.config.CONFIDENCE_THRESHOLD]
                moderate_confidence = [r for r in new_stock_recs if self.config.MIN_CONFIDENCE_FOR_TRADE <= r['confidence'] < self.config.CONFIDENCE_THRESHOLD]
                
                if high_confidence:
                    print(f"\n🟢 HIGH CONFIDENCE ({self.config.CONFIDENCE_THRESHOLD:.1%}+) - RECOMMENDED:")
                    for i, rec in enumerate(high_confidence, 1):
                        self._display_new_stock_recommendation(i, rec, "🟢 EXECUTE")
                
                if moderate_confidence:
                    print(f"\n🟡 MODERATE CONFIDENCE ({self.config.MIN_CONFIDENCE_FOR_TRADE:.1%}+) - REVIEW:")
                    start_idx = len(high_confidence) + 1
                    for i, rec in enumerate(moderate_confidence, start_idx):
                        self._display_new_stock_recommendation(i, rec, "🟡 REVIEW")
            
            # Summary statistics
            print("\n" + "="*100)
            print(f"📊 SUMMARY:")
            print(f"  Portfolio Stock Actions: {len(portfolio_recs)}")
            print(f"  New Stock Opportunities: {len(new_stock_recs)}")
            print(f"  Total Recommendations: {len(recommendations)}")
            
            if recommendations:
                avg_confidence = np.mean([r['confidence'] for r in recommendations])
                print(f"  Average Confidence: {avg_confidence:.1%}")
            
            print("="*100)
            print("🚀 Two-Stage Analysis: Advanced filtering for optimal stock selection")
        
        except Exception as e:
            logger.error(f"Error displaying recommendations: {e}")
    
    def _display_portfolio_stock_recommendation(self, index: int, rec: Dict) -> None:
        """Display recommendation for stocks already in portfolio"""
        try:
            action_emoji = "🔴" if rec['action'] == 'SELL' else "🟢" if rec['action'] == 'BUY' else "🟡"
            
            print(f"{index}. {rec['symbol']} - {action_emoji} {rec['action']}")
            print(f"   Current Position: {rec['current_position_size']:,} shares @ avg ${rec['avg_buy_price']:.2f}")
            print(f"   Current Price: ${rec['current_price']:.2f} | "
                  f"P/L: ${rec['unrealized_pnl']:+,.2f} ({rec['unrealized_pnl_percent']:+.1f}%)")
            print(f"   Recommended Sell Price: ${rec['sell_price']:.2f}")
            print(f"   Confidence: {rec['confidence']:.1%}")
            
            # Technical details
            tech_scores = rec.get('technical_scores', {})
            if tech_scores:
                rsi = tech_scores.get('rsi', 'N/A')
                macd = tech_scores.get('macd', 'N/A')
                print(f"   Technical: RSI={rsi} | MACD={macd}")
            
            # Reasoning
            reasoning = rec.get('reasoning', 'Portfolio position review')
            print(f"   💡 {reasoning}")
            print()
        
        except Exception as e:
            logger.error(f"Error displaying portfolio recommendation {index}: {e}")
    
    def _display_new_stock_recommendation(self, index: int, rec: Dict, status: str) -> None:
        """Display recommendation for new stock opportunities"""
        try:
            print(f"\n{index}. {rec['symbol']} - BUY | {status}")
            print(f"   Confidence: {rec['confidence']:.1%} | "
                  f"Price: ${rec['current_price']:.2f} | "
                  f"Target: ${rec.get('price_target', 0):.2f}")
            print(f"   Expected Return: {rec.get('expected_return', 0):.1%} | "
                  f"Stop Loss: ${rec['sell_price']:.2f}")
            
            # Technical details
            tech_scores = rec.get('technical_scores', {})
            if tech_scores:
                rsi = tech_scores.get('rsi', 'N/A')
                macd = tech_scores.get('macd', 'N/A')
                vol = tech_scores.get('volatility', 0)
                print(f"   Technical: RSI={rsi} | MACD={macd} | Vol={vol:.1%}")
            
            # Two-stage specific metrics
            ml_features = rec.get('ml_features', {})
            if ml_features:
                stage_score = ml_features.get('two_stage_score', 0)
                tech_strength = ml_features.get('technical_strength', 'Unknown')
                liquidity = ml_features.get('liquidity_grade', 'Unknown')
                print(f"   Two-Stage: Score={stage_score:.1f} | "
                      f"Technical={tech_strength} | "
                      f"Liquidity={liquidity}")
            
            # Reasoning (truncated)
            reasoning = rec.get('reasoning', 'Two-stage analysis selection')
            print(f"   💡 {reasoning[:100]}...")
        
        except Exception as e:
            logger.error(f"Error displaying recommendation {index}: {e}")
    
    def run_interactive_mode(self) -> None:
        """Interactive mode with two-stage system"""
        logger.info("🚀 Starting Two-Stage ML Trading System...")
        
        while True:
            try:
                print("\n" + "="*80)
                print("🎯 TWO-STAGE ML TRADING SYSTEM")
                print("="*80)
                print("1. 🚀 Two-Stage Daily Update (4000 → 50 → 10)")
                print("2. 📊 Portfolio Summary (with Live Prices)")
                print("3. 📝 Trade Logger")
                print("4. 🧠 System Information")
                print("5. 🎯 Test Two-Stage Kelly Sizing")
                print("6. 📈 Performance Analytics")
                print("7. ⚙️ Configuration")
                print("8. 🚪 Exit")
                
                choice = input("\nSelect option (1-8): ").strip()
                
                if choice == '1':
                    print("🚀 Running Two-Stage Analysis...")
                    asyncio.run(self.run_two_stage_daily_update())
                
                elif choice == '2':
                    portfolio = self.portfolio_manager.get_portfolio_summary()
                    self._display_enhanced_portfolio_summary(portfolio)
                
                elif choice == '3':
                    try:
                        from trade_logger import run_trade_logging_interface
                        run_trade_logging_interface()
                    except ImportError as e:
                        logger.error(f"Trade logger import error: {e}")
                        print("❌ Trade logger not available - please check trade_logger.py")
                
                elif choice == '4':
                    model_info = self.ml_engine.get_model_info()
                    self._display_system_info(model_info)
                
                elif choice == '5':
                    self._test_two_stage_kelly_sizing()
                
                elif choice == '6':
                    self._show_performance_analytics()
                
                elif choice == '7':
                    self._display_configuration()
                
                elif choice == '8':
                    logger.info("👋 Exiting Two-Stage ML Trading System...")
                    break
                
                else:
                    print("❌ Invalid option. Please select 1-8.")
            
            except KeyboardInterrupt:
                logger.info("👋 Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error: {e}")
    
    def _display_system_info(self, model_info: Dict) -> None:
        """Display system information"""
        print("\n" + "="*60)
        print("🧠 TWO-STAGE SYSTEM INFORMATION")
        print("="*60)
        
        for key, value in model_info.items():
            if isinstance(value, bool):
                status = "✅ Enabled" if value else "❌ Disabled"
                print(f"  {key}: {status}")
            else:
                print(f"  {key}: {value}")
    
    def _test_two_stage_kelly_sizing(self) -> None:
        """Test Kelly sizing with two-stage recommendations"""
        print("\n🎯 TWO-STAGE KELLY SIZING TEST")
        print("="*60)
        
        portfolio_value = self.portfolio_manager.get_portfolio_summary()['total_value']
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Kelly Threshold: {self.config.CONFIDENCE_THRESHOLD:.1%}")
        
        test_cases = [
            {"price": 50.0, "expected_return": 0.15, "confidence": 0.92, "name": "Two-stage top pick"},
            {"price": 25.0, "expected_return": 0.12, "confidence": 0.89, "name": "Kelly threshold"},
            {"price": 100.0, "expected_return": 0.10, "confidence": 0.85, "name": "Moderate confidence"},
            {"price": 75.0, "expected_return": 0.08, "confidence": 0.70, "name": "Below threshold"},
        ]
        
        for case in test_cases:
            print(f"\n📊 {case['name']}:")
            print(f"  Price: ${case['price']:.2f}")
            print(f"  Expected Return: {case['expected_return']:.1%}")
            print(f"  Confidence: {case['confidence']:.1%}")
            
            position_size = self.portfolio_manager.get_enhanced_kelly_position_size(
                case['price'], case['expected_return'], case['confidence'], portfolio_value
            )
            
            if position_size > 0:
                quantity = int(position_size // case['price'])
                if quantity > 0:
                    total = quantity * case['price']
                    pct = (total / portfolio_value) * 100
                    meets_kelly = case['confidence'] >= self.config.CONFIDENCE_THRESHOLD
                    status = "✅ EXECUTE" if meets_kelly else "⚠️ REVIEW"
                    print(f"  {status}: {quantity} shares = ${total:.2f} ({pct:.1f}%)")
                else:
                    print(f"  ⚠️ Position too small")
            else:
                print(f"  ❌ No position recommended")
    
    def _show_performance_analytics(self) -> None:
        """Show performance analytics"""
        print("\n📈 PERFORMANCE ANALYTICS")
        print("="*60)
        print("🚧 Two-stage performance tracking:")
        print("  • Stage 1 filtering efficiency")
        print("  • Stage 2 selection accuracy")
        print("  • Kelly sizing optimization")
        print("  • Bayesian learning progress")
        print("  • Live price tracking accuracy")
    
    def _display_configuration(self) -> None:
        """Display current configuration"""
        print("\n⚙️ TWO-STAGE SYSTEM CONFIGURATION")
        print("="*60)
        print(f"Initial Capital: ${self.config.INITIAL_CAPITAL:,.2f}")
        print(f"Kelly Fraction: {self.config.KELLY_FRACTION}")
        print(f"Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD:.1%}")
        print(f"Min Confidence: {self.config.MIN_CONFIDENCE_FOR_TRADE:.1%}")
        print(f"Max Position Size: {self.config.MAX_POSITION_SIZE:.1%}")
        print(f"Max Daily Loss: {getattr(self.config, 'MAX_DAILY_LOSS', 0.05):.1%}")
        print(f"Stop Loss Percentage: {getattr(self.config, 'STOP_LOSS_PERCENTAGE', 0.15):.1%}")


def main():
    """Main entry point for two-stage trading system"""
    try:
        print("🎯 Initializing Two-Stage ML Trading System...")
        print("📊 Analysis Flow: 4000+ stocks → Top 50 → Final recommendations")
        print("💹 Live price tracking enabled via Yahoo Finance")
        
        trading_system = TwoStageMLTradingSystem()
        trading_system.run_interactive_mode()
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
