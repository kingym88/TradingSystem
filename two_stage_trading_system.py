"""
Two-Stage Enhanced ML Trading System 
UPDATED: Handles 4000 → 50 → 10 intelligent stock filtering with optimized performance
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

class TwoStagePortfolioManager:
    """Portfolio management optimized for two-stage analysis"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.config = config
        
    def get_enhanced_kelly_position_size(self, price: float, expected_return: float, 
                                       confidence: float, portfolio_value: float) -> float:
        """Enhanced Kelly sizing with two-stage optimization"""
        try:
            if expected_return <= 0 or confidence <= 0 or confidence >= 1:
                return 0.0

            # Kelly threshold calculation
            kelly_threshold = 1 / (1 + expected_return)
            
            # Two-stage confidence adjustment
            if confidence >= kelly_threshold:
                # Full Kelly with two-stage boost
                kelly_f = (expected_return * confidence - (1 - confidence)) / expected_return
                position_fraction = kelly_f * self.config.KELLY_FRACTION
                
                # Two-stage bonus for high-quality selections
                if confidence >= 0.9:  # Top-tier from two-stage
                    position_fraction *= 1.1  # 10% bonus
                
                position_size = portfolio_value * min(position_fraction, self.config.MAX_POSITION_SIZE)
                logger.info(f"Two-Stage Kelly: {position_fraction:.1%} of portfolio")
            else:
                # Enhanced simplified sizing
                simplified_fraction = confidence * expected_return * 3.0
                position_size = portfolio_value * min(simplified_fraction, self.config.MAX_POSITION_SIZE * 0.8)
                logger.info(f"Two-Stage Simplified: {simplified_fraction:.1%} of portfolio")

            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error in two-stage Kelly sizing: {e}")
            return 0.0
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            trades = self.db_manager.get_all_trades()
            if not trades:
                return {
                    'total_value': self.config.INITIAL_CAPITAL,
                    'cash': self.config.INITIAL_CAPITAL,
                    'invested_amount': 0.0,
                    'holdings': {},
                    'total_return': 0.0,
                    'num_positions': 0
                }

            # Calculate portfolio from all trades
            cash = self.config.INITIAL_CAPITAL
            holdings = {}

            for trade in sorted(trades, key=lambda x: x.get('timestamp', datetime.min)):
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                price = trade['price']
                total_amount = quantity * price

                if action == 'BUY':
                    cash -= total_amount
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                elif action == 'SELL':
                    cash += total_amount
                    holdings[symbol] = holdings.get(symbol, 0) - quantity
                    if holdings[symbol] <= 0:
                        holdings.pop(symbol, None)

            # Calculate current market value (simplified)
            invested_amount = 0.0
            for symbol, qty in holdings.items():
                if qty > 0:
                    # Use last trade price as approximation
                    last_price = next((t['price'] for t in reversed(trades) if t['symbol'] == symbol), 0)
                    invested_amount += qty * last_price

            total_value = cash + invested_amount
            total_return = ((total_value - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL) * 100

            return {
                'total_value': total_value,
                'cash': cash,
                'invested_amount': invested_amount,
                'holdings': {k: v for k, v in holdings.items() if v > 0},
                'total_return': total_return,
                'num_positions': len([k for k, v in holdings.items() if v > 0])
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio: {e}")
            return {
                'total_value': self.config.INITIAL_CAPITAL,
                'cash': self.config.INITIAL_CAPITAL,
                'invested_amount': 0.0,
                'holdings': {},
                'total_return': 0.0,
                'num_positions': 0
            }

class TwoStageMLTradingSystem:
    """
    Main trading system with two-stage stock analysis (4000 → 50 → 10)
    """

    def __init__(self):
        self.config = config
        self.portfolio_manager = TwoStagePortfolioManager()
        self.data_manager = get_two_stage_data_manager()
        self.ml_engine = get_two_stage_ml_engine()
        self.db_manager = get_db_manager()

        logger.info("🚀 Two-Stage ML Trading System initialized (4000 → 50 → 10)")

    async def run_two_stage_daily_update(self) -> None:
        """Run enhanced daily update with two-stage analysis"""
        try:
            start_time = time.time()
            logger.info("🎯 Starting Two-Stage Daily Update (4000 → 50 → 10)...")

            # Get current portfolio
            portfolio_data = self.portfolio_manager.get_portfolio_summary()
            logger.info(f"📊 Portfolio: ${portfolio_data['total_value']:.2f} "
                       f"(Return: {portfolio_data['total_return']:.2f}%)")

            # Learn from recent trades
            logger.info("🧠 Learning from trade history...")
            self.ml_engine.learn_from_trades()

            # Generate two-stage recommendations
            logger.info("🎯 Running Two-Stage Analysis...")
            recommendations = await self.ml_engine.generate_recommendations()

            if not recommendations:
                logger.warning("❌ No recommendations generated from two-stage analysis")
                return

            # Display recommendations
            self._display_two_stage_recommendations(recommendations, portfolio_data)

            # Simulate trade execution for high-confidence recommendations
            executed_trades = 0
            for rec in recommendations:
                try:
                    if rec['confidence'] >= self.config.CONFIDENCE_THRESHOLD:
                        # Simulate trade execution
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
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
            logger.info(f"📊 Processed: 4000+ → 50 → {len(recommendations)} recommendations")
            logger.info(f"🎯 High-confidence trades: {executed_trades}")

        except Exception as e:
            logger.error(f"Error in two-stage daily update: {e}")

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

    def _display_two_stage_recommendations(self, recommendations: List[Dict], portfolio_data: Dict) -> None:
        """Display two-stage analysis results"""
        try:
            print("\n" + "="*100)
            print("🎯 TWO-STAGE ML TRADING SYSTEM - INTELLIGENT STOCK SELECTION")
            print("="*100)
            print("📊 ANALYSIS FLOW: 4000+ stocks → Top 50 candidates → Final 10 recommendations")

            # Portfolio summary
            print(f"\n💰 PORTFOLIO STATUS:")
            print(f"   Total Value: ${portfolio_data['total_value']:.2f}")
            print(f"   Cash Available: ${portfolio_data['cash']:.2f}")
            print(f"   Total Return: {portfolio_data['total_return']:.2f}%")
            print(f"   Active Positions: {portfolio_data['num_positions']}")

            if not recommendations:
                print("\n❌ No recommendations generated")
                return

            # Categorize recommendations
            kelly_threshold_met = []
            moderate_confidence = []
            
            for rec in recommendations:
                if rec['confidence'] >= self.config.CONFIDENCE_THRESHOLD:
                    kelly_threshold_met.append(rec)
                elif rec['confidence'] >= self.config.MIN_CONFIDENCE_FOR_TRADE:
                    moderate_confidence.append(rec)

            print(f"\n🎯 TWO-STAGE RECOMMENDATIONS:")
            print("-" * 100)

            # Display high-confidence recommendations
            if kelly_threshold_met:
                print(f"\n🟢 KELLY THRESHOLD MET ({self.config.CONFIDENCE_THRESHOLD:.1%}+) - AUTO-EXECUTE:")
                for i, rec in enumerate(kelly_threshold_met, 1):
                    self._display_single_recommendation(i, rec, "🟢 EXECUTE")

            # Display moderate-confidence recommendations
            if moderate_confidence:
                print(f"\n🟡 MODERATE CONFIDENCE ({self.config.MIN_CONFIDENCE_FOR_TRADE:.1%}+) - MANUAL REVIEW:")
                start_idx = len(kelly_threshold_met) + 1
                for i, rec in enumerate(moderate_confidence, start_idx):
                    self._display_single_recommendation(i, rec, "🟡 REVIEW")

            # Summary statistics
            print("\n" + "-" * 100)
            print(f"📊 SUMMARY:")
            print(f"   🟢 Kelly Threshold Met: {len(kelly_threshold_met)} (auto-execute)")
            print(f"   🟡 Moderate Confidence: {len(moderate_confidence)} (manual review)")
            print(f"   📈 Total Recommendations: {len(recommendations)}")
            
            # Two-stage performance metrics
            if recommendations:
                avg_confidence = np.mean([r['confidence'] for r in recommendations])
                avg_expected_return = np.mean([r.get('expected_return', 0) for r in recommendations])
                print(f"   🎯 Average Confidence: {avg_confidence:.1%}")
                print(f"   📈 Average Expected Return: {avg_expected_return:.1%}")

            print("="*100)
            print("🚀 Two-Stage Analysis: Advanced filtering for optimal stock selection")

        except Exception as e:
            logger.error(f"Error displaying recommendations: {e}")

    def _display_single_recommendation(self, index: int, rec: Dict, status: str) -> None:
        """Display a single recommendation with detailed info"""
        try:
            print(f"\n{index}. {rec['symbol']} - {rec['action']} | {status}")
            print(f"   Confidence: {rec['confidence']:.1%} | "
                  f"Price: ${rec['current_price']:.2f} | "
                  f"Target: ${rec.get('price_target', 0):.2f}")
            print(f"   Expected Return: {rec.get('expected_return', 0):.1%} | "
                  f"Risk Score: {rec.get('risk_score', 0):.1%}")
            
            # Technical details
            tech_scores = rec.get('technical_scores', {})
            if tech_scores:
                rsi = tech_scores.get('rsi', 'N/A')
                macd = tech_scores.get('macd', 'N/A') 
                adx = tech_scores.get('adx', 'N/A')
                vol = tech_scores.get('volatility', 0)
                print(f"   Technical: RSI={rsi} | MACD={macd} | ADX={adx} | Vol={vol:.1%}")
            
            # Two-stage specific metrics
            ml_features = rec.get('ml_features', {})
            if ml_features:
                stage_score = ml_features.get('two_stage_score', 0)
                tech_strength = ml_features.get('technical_strength', 'Unknown')
                liquidity = ml_features.get('liquidity_grade', 'Unknown')
                momentum = ml_features.get('momentum_grade', 'Unknown')
                print(f"   Two-Stage: Score={stage_score:.1f} | "
                      f"Technical={tech_strength} | "
                      f"Liquidity={liquidity} | "
                      f"Momentum={momentum}")
            
            # Enhanced metrics
            if rec.get('bayesian_confidence'):
                bayesian_conf = rec['bayesian_confidence']
                similar_trades = rec.get('similar_trades_count', 0)
                print(f"   Bayesian: {bayesian_conf:.1%} (from {similar_trades} similar trades)")
            
            # Reasoning (truncated)
            reasoning = rec.get('reasoning', 'Two-stage analysis selection')
            print(f"   Reasoning: {reasoning[:80]}...")
            
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
                print("2. 📊 Portfolio Summary")
                print("3. 📝 Trade Logger")
                print("4. 🧠 System Information")
                print("5. 🎯 Test Two-Stage Kelly Sizing")
                print("6. 📈 Performance Analytics")
                print("7. ⚙️  Configuration")
                print("8. 🚪 Exit")

                choice = input("\nSelect option (1-8): ").strip()

                if choice == '1':
                    print("🚀 Running Two-Stage Analysis...")
                    asyncio.run(self.run_two_stage_daily_update())
                
                elif choice == '2':
                    portfolio = self.portfolio_manager.get_portfolio_summary()
                    self._display_portfolio_summary(portfolio)
                
                elif choice == '3':
                    try:
                        from trade_logger import run_trade_logging_interface
                        run_trade_logging_interface()
                    except ImportError:
                        print("Trade logger not available")
                
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

    def _display_portfolio_summary(self, portfolio: Dict) -> None:
        """Display portfolio summary"""
        print("\n" + "="*60)
        print("📊 PORTFOLIO SUMMARY")
        print("="*60)
        print(f"💰 Total Value: ${portfolio['total_value']:.2f}")
        print(f"💵 Cash: ${portfolio['cash']:.2f}")
        print(f"📈 Invested: ${portfolio['invested_amount']:.2f}")
        print(f"🎯 Return: {portfolio['total_return']:.2f}%")
        print(f"📋 Positions: {portfolio['num_positions']}")

        if portfolio['holdings']:
            print(f"\n📋 CURRENT HOLDINGS:")
            for symbol, quantity in portfolio['holdings'].items():
                print(f"   {symbol}: {quantity:,} shares")

    def _display_system_info(self, model_info: Dict) -> None:
        """Display system information"""
        print("\n" + "="*60)
        print("🧠 TWO-STAGE SYSTEM INFORMATION")
        print("="*60)
        
        for key, value in model_info.items():
            if isinstance(value, bool):
                status = "✅ Enabled" if value else "❌ Disabled"
                print(f"   {key}: {status}")
            else:
                print(f"   {key}: {value}")

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
            print(f"   Price: ${case['price']:.2f}")
            print(f"   Expected Return: {case['expected_return']:.1%}")
            print(f"   Confidence: {case['confidence']:.1%}")

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
                    print(f"   {status}: {quantity} shares = ${total:.2f} ({pct:.1f}%)")
                else:
                    print(f"   ⚠️ Position too small")
            else:
                print(f"   ❌ No position recommended")

    def _show_performance_analytics(self) -> None:
        """Show performance analytics"""
        print("\n📈 PERFORMANCE ANALYTICS")
        print("="*60)
        print("🚧 Two-stage performance tracking:")
        print("   • Stage 1 filtering efficiency")
        print("   • Stage 2 selection accuracy") 
        print("   • Kelly sizing optimization")
        print("   • Bayesian learning progress")

    def _display_configuration(self) -> None:
        """Display current configuration"""
        print("\n⚙️ TWO-STAGE SYSTEM CONFIGURATION")
        print("="*60)
        print(f"Initial Capital: ${self.config.INITIAL_CAPITAL:,.2f}")
        print(f"Kelly Fraction: {self.config.KELLY_FRACTION}")
        print(f"Confidence Threshold: {self.config.CONFIDENCE_THRESHOLD:.1%}")
        print(f"Min Confidence: {self.config.MIN_CONFIDENCE_FOR_TRADE:.1%}")
        print(f"Max Position Size: {self.config.MAX_POSITION_SIZE:.1%}")

def main():
    """Main entry point for two-stage trading system"""
    try:
        print("🎯 Initializing Two-Stage ML Trading System...")
        print("📊 Analysis Flow: 4000+ stocks → Top 50 → Final 10 recommendations")
        
        trading_system = TwoStageMLTradingSystem()
        trading_system.run_interactive_mode()

    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"❌ Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()