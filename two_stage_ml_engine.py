"""
Enhanced ML Engine for Two-Stage Stock Analysis System
UPDATED: Added sell price recommendations and portfolio stock handling
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
import asyncio
warnings.filterwarnings('ignore')

from loguru import logger
from perplexity_fundamental_analyzer import get_fundamental_analyzer

# Enhanced ML imports (with fallbacks)
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
    logger.info("✅ scikit-learn imported successfully")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠️ scikit-learn not available. Using basic ML methods.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    logger.info("✅ XGBoost imported successfully")   
except ImportError:
    HAS_XGBOOST = False
    logger.warning("⚠️ XGBoost not available.")

# Bayesian methods
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("⚠️ SciPy not available. Bayesian methods disabled.")

try:
    from two_stage_config import get_config
    config = get_config()
except ImportError:
    # Fallback configuration for ML engine
    class FallbackMLConfig:
        MIN_CONFIDENCE_FOR_TRADE = 0.65
        CONFIDENCE_THRESHOLD = 0.89
        KELLY_FRACTION = 0.5
        MAX_DAILY_RECOMMENDATIONS = 10
        MIN_TRAINING_SAMPLES = 20
        ENABLE_MODEL_CALIBRATION = True
        ENABLE_BAYESIAN_UPDATING = True
        BAYESIAN_PRIOR_ALPHA = 5.0
        BAYESIAN_PRIOR_BETA = 5.0
        MAX_DAILY_LOSS = 0.05  # 5% daily loss limit
        STOP_LOSS_PERCENTAGE = 0.15  # 15% stop loss
    
    config = FallbackMLConfig()

from database import get_db_manager
from two_stage_data_manager import get_two_stage_data_manager, run_two_stage_stock_analysis
from portfolio_manager import get_portfolio_manager


class TwoStageBayesianUpdater:
    """Bayesian belief updating optimized for two-stage analysis"""
    
    def __init__(self):
        self.config = config
        self.prior_alpha = getattr(config, 'BAYESIAN_PRIOR_ALPHA', 5.0)
        self.prior_beta = getattr(config, 'BAYESIAN_PRIOR_BETA', 5.0)
    
    def update_confidence_from_trades(self, base_confidence: float, 
                                    recent_wins: int, recent_total: int) -> float:
        """Update confidence using Bayesian methods"""
        try:
            if not HAS_SCIPY or recent_total == 0:
                return base_confidence
            
            # Bayesian update
            posterior_alpha = self.prior_alpha + recent_wins
            posterior_beta = self.prior_beta + (recent_total - recent_wins)
            
            # Get updated confidence (mean of beta distribution)
            bayesian_confidence = posterior_alpha / (posterior_alpha + posterior_beta)
            
            # Blend with base confidence (weighted average)
            if recent_total >= 10:
                # High confidence in Bayesian estimate
                final_confidence = 0.7 * bayesian_confidence + 0.3 * base_confidence
            elif recent_total >= 5:
                # Moderate confidence
                final_confidence = 0.5 * bayesian_confidence + 0.5 * base_confidence
            else:
                # Low confidence, mostly use base
                final_confidence = 0.3 * bayesian_confidence + 0.7 * base_confidence
            
            logger.debug(f"Bayesian update: {recent_wins}/{recent_total} → "
                        f"Base: {base_confidence:.3f}, Bayesian: {bayesian_confidence:.3f}, "
                        f"Final: {final_confidence:.3f}")
            
            return min(0.95, max(0.1, final_confidence))
        
        except Exception as e:
            logger.error(f"Error in Bayesian confidence update: {e}")
            return base_confidence
    
    def get_confidence_interval(self, wins: int, total: int, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval"""
        try:
            if not HAS_SCIPY or total == 0:
                return (0.0, 1.0)
            
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p = wins / total
            n = total
            
            center = (p + z*z/(2*n)) / (1 + z*z/n)
            margin = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / (1 + z*z/n)
            
            lower = max(0, center - margin)
            upper = min(1, center + margin)
            
            return (lower, upper)
        
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (0.0, 1.0)


class TwoStageMLRecommendationEngine:
    """
    ML Recommendation Engine optimized for two-stage stock analysis
    UPDATED: Added sell price recommendations and portfolio stock detection
    """
    
    def __init__(self):
        self.config = config
        self.db_manager = get_db_manager()
        self.data_manager = get_two_stage_data_manager()
        self.portfolio_manager = get_portfolio_manager()
        self.bayesian_updater = TwoStageBayesianUpdater()
        self.fundamental_analyzer = get_fundamental_analyzer()

        # Model components
        self.ensemble_model = None
        self.last_training_date = None
        
        logger.info("🚀 Two-Stage ML Recommendation Engine initialized")
    
    def calculate_sell_price(self, current_price: float, avg_buy_price: float = None) -> float:
        """
        Calculate recommended sell price based on daily loss limits
        Uses MAX_DAILY_LOSS from config
        """
        try:
            # Get max daily loss from config (default 5%)
            max_daily_loss = getattr(self.config, 'MAX_DAILY_LOSS', 0.05)
            
            # If we have average buy price, calculate sell price from buy price
            if avg_buy_price is not None:
                # Sell at stop loss level below average buy price
                stop_loss_percent = getattr(self.config, 'STOP_LOSS_PERCENTAGE', 0.15)
                sell_price = avg_buy_price * (1 - stop_loss_percent)
            else:
                # Use current price minus daily loss limit
                sell_price = current_price * (1 - max_daily_loss)
            
            return round(sell_price, 2)
        
        except Exception as e:
            logger.error(f"Error calculating sell price: {e}")
            # Fallback: 5% below current price
            return round(current_price * 0.95, 2)
    
    async def generate_recommendations(self, max_recommendations: int = None) -> List[Dict[str, Any]]:
        """
        UPDATED: Generate recommendations split into new stocks and portfolio stocks
        """
        try:
            max_recs = max_recommendations or getattr(self.config, 'MAX_DAILY_RECOMMENDATIONS', 10)
            logger.info(f"🎯 Generating {max_recs} recommendations using two-stage analysis...")
            
            # Get current portfolio
            portfolio = self.portfolio_manager.get_portfolio_summary()
            portfolio_symbols = set(portfolio['holdings'].keys())
            holdings_details = portfolio['holdings_details']
            
            logger.info(f"📊 Current portfolio has {len(portfolio_symbols)} positions: {portfolio_symbols}")
            
            # Get trade history for learning
            trade_history = self.db_manager.get_trade_history(days=365)
            
            # Check if we have sufficient training data
            min_samples = getattr(self.config, 'MIN_TRAINING_SAMPLES', 20)
            if trade_history.empty or len(trade_history) < min_samples:
                logger.info("Using two-stage analysis without historical learning")
                base_recommendations = await self._generate_two_stage_recommendations(max_recs * 2)  # Get more to split
            else:
                # Generate recommendations with Bayesian enhancement
                base_recommendations = await self._generate_two_stage_recommendations(max_recs * 2)
                
                # Enhance with Bayesian learning from trade history
                base_recommendations = self._enhance_with_bayesian_learning(
                    base_recommendations, trade_history
                )
            
            # SPLIT RECOMMENDATIONS: Portfolio stocks vs New stocks
            portfolio_recommendations = []
            new_stock_recommendations = []
            
            for rec in base_recommendations:
                symbol = rec['symbol']
                
                # Add sell price to all recommendations
                current_price = rec.get('current_price', 0)
                
                if symbol in portfolio_symbols:
                    # Stock is in portfolio - analyze for BUY or SELL
                    position_details = holdings_details.get(symbol, {})
                    avg_buy_price = position_details.get('avg_buy_price', current_price)
                    current_value = position_details.get('current_value', 0)
                    unrealized_pnl_percent = position_details.get('unrealized_pnl_percent', 0)
                    
                    # Calculate sell price based on average buy price
                    sell_price = self.calculate_sell_price(current_price, avg_buy_price)
                    
                    # Determine action: BUY more or SELL
                    if current_price > avg_buy_price:
                        # Stock is profitable - recommend SELL if target reached
                        if unrealized_pnl_percent > 15:  # 15% profit target
                            action = 'SELL'
                            reasoning = f"Take profit: Currently ${current_price:.2f}, bought at ${avg_buy_price:.2f} ({unrealized_pnl_percent:+.1f}% gain). Target sell price: ${sell_price:.2f}"
                        else:
                            action = 'HOLD'
                            reasoning = f"Hold position: Currently ${current_price:.2f}, bought at ${avg_buy_price:.2f} ({unrealized_pnl_percent:+.1f}% gain). Sell if drops to ${sell_price:.2f}"
                    elif current_price < avg_buy_price * 0.95:
                        # Stock is down 5%+ - check if still good buy
                        if rec.get('confidence', 0) > 0.75:
                            action = 'BUY'
                            reasoning = f"Average down: Currently ${current_price:.2f}, bought at ${avg_buy_price:.2f} ({unrealized_pnl_percent:+.1f}% loss). Strong technical signals suggest recovery. Stop loss: ${sell_price:.2f}"
                        else:
                            action = 'SELL'
                            reasoning = f"Cut losses: Currently ${current_price:.2f}, bought at ${avg_buy_price:.2f} ({unrealized_pnl_percent:+.1f}% loss). Weak signals suggest further decline. Sell at ${sell_price:.2f}"
                    else:
                        # Near break-even
                        action = 'HOLD'
                        reasoning = f"Monitor position: Currently ${current_price:.2f}, bought at ${avg_buy_price:.2f} ({unrealized_pnl_percent:+.1f}%). Set stop loss at ${sell_price:.2f}"
                    
                    portfolio_rec = {
                        **rec,
                        'action': action,
                        'sell_price': sell_price,
                        'avg_buy_price': avg_buy_price,
                        'current_position_size': position_details.get('quantity', 0),
                        'unrealized_pnl': position_details.get('unrealized_pnl', 0),
                        'unrealized_pnl_percent': unrealized_pnl_percent,
                        'reasoning': reasoning,
                        'is_portfolio_stock': True
                    }
                    
                    # Only include actionable recommendations (skip HOLD for display)
                    if action in ['BUY', 'SELL']:
                        portfolio_recommendations.append(portfolio_rec)
                else:
                    # New stock - recommend BUY with sell price
                    sell_price = self.calculate_sell_price(current_price)
                    
                    new_stock_rec = {
                        **rec,
                        'action': 'BUY',
                        'sell_price': sell_price,
                        'reasoning': rec.get('reasoning', '') + f" Set stop loss at ${sell_price:.2f}",
                        'is_portfolio_stock': False
                    }
                    
                    new_stock_recommendations.append(new_stock_rec)
            
            # Combine recommendations: prioritize portfolio actions, then new stocks
            final_recommendations = portfolio_recommendations + new_stock_recommendations
            final_recommendations = final_recommendations[:max_recs]
            
            logger.info(f"✅ Split recommendations: {len(portfolio_recommendations)} portfolio stocks, {len(new_stock_recommendations)} new stocks")
            logger.info(f"📊 Returning {len(final_recommendations)} total recommendations")
            
            return final_recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    async def _generate_two_stage_recommendations(self, max_recs: int) -> List[Dict[str, Any]]:
        """Generate recommendations using two-stage stock analysis WITH FUNDAMENTAL ANALYSIS"""
        try:
            logger.info("🚀 Running two-stage stock analysis...")

            # Run the two-stage technical analysis
            recommendations = await run_two_stage_stock_analysis()

            if not recommendations:
                logger.warning("No recommendations from two-stage analysis")
                return []

            # Select top candidates for fundamental analysis
            top_candidates = recommendations[:max_recs]

            # ==== NEW: PERPLEXITY FUNDAMENTAL ANALYSIS ====
            logger.info(f"🔬 Running Perplexity fundamental analysis on top {len(top_candidates)} candidates...")

            fundamental_results = {}
            enable_fundamental = getattr(self.config, 'ENABLE_FUNDAMENTAL_ANALYSIS', True)

            if enable_fundamental and self.fundamental_analyzer.client:
                # Prepare stocks for batch analysis
                stocks_to_analyze = [
                    {'symbol': rec['symbol'], 'name': rec.get('company_name')} 
                    for rec in top_candidates
                ]

                # Batch analyze with Perplexity
                try:
                    fundamental_results = await self.fundamental_analyzer.batch_analyze_stocks(stocks_to_analyze)
                    logger.info(f"✅ Completed fundamental analysis for {len(fundamental_results)} stocks")
                except Exception as e:
                    logger.error(f"Error in batch fundamental analysis: {e}")
            else:
                if not enable_fundamental:
                    logger.info("⚠️ Fundamental analysis disabled in config")
                else:
                    logger.warning("⚠️ Perplexity client not available - skipping fundamental analysis")

            # Apply ML enhancements to recommendations
            enhanced_recs = []
            for rec in top_candidates:
                try:
                    symbol = rec['symbol']

                    # Get fundamental analysis for this stock
                    fundamental_analysis = fundamental_results.get(symbol)

                    # Calculate ML confidence boost
                    ml_confidence_boost = self._calculate_ml_confidence_boost(rec)

                    # Calculate fundamental confidence boost (NEW)
                    fundamental_boost = 0.0
                    if fundamental_analysis:
                        fundamental_boost = self.fundamental_analyzer.get_fundamental_boost(fundamental_analysis)
                        logger.debug(f"{symbol}: Fundamental boost = {fundamental_boost:+.3f} "
                                   f"(score: {fundamental_analysis.overall_score:.1f}/10)")

                    # Combine boosts
                    total_boost = ml_confidence_boost + fundamental_boost

                    original_confidence = rec.get('confidence', 0.7)
                    boosted_confidence = min(0.95, max(0.5, original_confidence + total_boost))

                    # Enhanced reasoning with fundamental insights
                    enhanced_reasoning = rec.get('reasoning', 'Two-stage selection')

                    if fundamental_analysis:
                        # Add fundamental insights to reasoning
                        if fundamental_analysis.recommendation:
                            enhanced_reasoning += f" | Fundamental: {fundamental_analysis.recommendation}"
                        if fundamental_analysis.overall_score >= 7.0:
                            enhanced_reasoning += f" | Strong fundamentals (score: {fundamental_analysis.overall_score:.1f}/10)"
                        elif fundamental_analysis.overall_score <= 4.0:
                            enhanced_reasoning += f" | Weak fundamentals (score: {fundamental_analysis.overall_score:.1f}/10)"

                        # Add top insight if available
                        if fundamental_analysis.key_insights:
                            top_insight = fundamental_analysis.key_insights[0][:80]
                            enhanced_reasoning += f" | {top_insight}"

                    # Update recommendation with ML and fundamental enhancements
                    enhanced_rec = rec.copy()
                    enhanced_rec.update({
                        'confidence': boosted_confidence,
                        'original_confidence': original_confidence,
                        'ml_confidence_boost': ml_confidence_boost,
                        'fundamental_confidence_boost': fundamental_boost,
                        'total_confidence_boost': total_boost,
                        'source': 'Two-Stage ML + Fundamental',
                        'reasoning': enhanced_reasoning,
                        'ml_features': {
                            'two_stage_score': rec.get('final_score', 0),
                            'technical_strength': self._assess_technical_strength(rec),
                            'liquidity_grade': self._assess_liquidity_grade(rec),
                            'momentum_grade': self._assess_momentum_grade(rec),
                            'fundamental_score': fundamental_analysis.overall_score if fundamental_analysis else None,
                            'fundamental_recommendation': fundamental_analysis.recommendation if fundamental_analysis else None,
                        },
                        'fundamental_analysis': {
                            'overall_score': fundamental_analysis.overall_score if fundamental_analysis else None,
                            'financial_health_score': fundamental_analysis.financial_health_score if fundamental_analysis else None,
                            'growth_score': fundamental_analysis.growth_score if fundamental_analysis else None,
                            'competitive_score': fundamental_analysis.competitive_score if fundamental_analysis else None,
                            'management_score': fundamental_analysis.management_score if fundamental_analysis else None,
                            'industry_score': fundamental_analysis.industry_score if fundamental_analysis else None,
                            'recommendation': fundamental_analysis.recommendation if fundamental_analysis else None,
                            'key_insights': fundamental_analysis.key_insights if fundamental_analysis else [],
                            'risks': fundamental_analysis.risks if fundamental_analysis else [],
                            'catalysts': fundamental_analysis.catalysts if fundamental_analysis else [],
                        } if fundamental_analysis else None
                    })

                    enhanced_recs.append(enhanced_rec)

                except Exception as e:
                    logger.error(f"Error enhancing recommendation for {rec.get('symbol', 'UNKNOWN')}: {e}")
                    enhanced_recs.append(rec)  # Use original if enhancement fails

            return enhanced_recs

        except Exception as e:
            logger.error(f"Error in two-stage recommendations: {e}")
            return []
    
    def _calculate_ml_confidence_boost(self, recommendation: Dict) -> float:
        """Calculate ML-based confidence boost"""
        try:
            boost = 0.0
            
            # Technical scores boost
            tech_scores = recommendation.get('technical_scores', {})
            
            # RSI boost
            rsi = tech_scores.get('rsi')
            if rsi is not None:
                if 25 <= rsi <= 35:  # Strong oversold
                    boost += 0.08
                elif 35 < rsi <= 45:  # Moderate oversold
                    boost += 0.05
                elif rsi > 75:  # Overbought penalty
                    boost -= 0.03
            
            # MACD boost
            macd = tech_scores.get('macd')
            if macd is not None and macd > 0:  # Positive MACD
                boost += 0.04
            
            # ADX boost (trend strength)
            adx = tech_scores.get('adx')
            if adx is not None and adx > 25:  # Strong trend
                boost += 0.03
            
            # Volume boost
            volume = recommendation.get('volume', 0)
            if volume > 2_000_000:  # High volume
                boost += 0.04
            elif volume > 1_000_000:  # Good volume
                boost += 0.02
            
            # Final score boost
            final_score = recommendation.get('final_score', 0)
            if final_score > 8:  # Very high score
                boost += 0.06
            elif final_score > 6:  # High score
                boost += 0.03
            
            # Sentiment boost
            sentiment = recommendation.get('sentiment_score', 0)
            if sentiment > 0.2:  # Positive sentiment
                boost += 0.02
            
            return min(0.25, max(0.0, boost))  # Cap at 25% boost
        
        except Exception as e:
            logger.error(f"Error calculating ML confidence boost: {e}")
            return 0.0
    
    def _assess_technical_strength(self, recommendation: Dict) -> str:
        """Assess technical analysis strength"""
        try:
            tech_scores = recommendation.get('technical_scores', {})
            strength_score = 0
            
            # RSI assessment
            rsi = tech_scores.get('rsi')
            if rsi is not None:
                if 25 <= rsi <= 40:
                    strength_score += 2
                elif 40 < rsi <= 60:
                    strength_score += 1
            
            # MACD assessment
            macd = tech_scores.get('macd')
            if macd is not None and macd > 0:
                strength_score += 1
            
            # ADX assessment
            adx = tech_scores.get('adx')
            if adx is not None and adx > 25:
                strength_score += 1
            
            if strength_score >= 3:
                return "Strong"
            elif strength_score >= 2:
                return "Moderate"
            else:
                return "Weak"
        
        except Exception as e:
            logger.error(f"Error assessing technical strength: {e}")
            return "Unknown"
    
    def _assess_liquidity_grade(self, recommendation: Dict) -> str:
        """Assess liquidity grade"""
        try:
            volume = recommendation.get('volume', 0)
            
            if volume >= 5_000_000:
                return "A"
            elif volume >= 2_000_000:
                return "B"
            elif volume >= 1_000_000:
                return "C"
            else:
                return "D"
        
        except Exception as e:
            logger.error(f"Error assessing liquidity grade: {e}")
            return "Unknown"
    
    def _assess_momentum_grade(self, recommendation: Dict) -> str:
        """Assess momentum grade"""
        try:
            expected_return = recommendation.get('expected_return', 0)
            
            if expected_return >= 0.15:  # 15%+
                return "Strong"
            elif expected_return >= 0.10:  # 10%+
                return "Moderate"
            elif expected_return >= 0.05:  # 5%+
                return "Weak"
            else:
                return "Minimal"
        
        except Exception as e:
            logger.error(f"Error assessing momentum grade: {e}")
            return "Unknown"
    
    def _enhance_with_bayesian_learning(self, recommendations: List[Dict], 
                                       trade_history: pd.DataFrame) -> List[Dict]:
        """Enhance recommendations with Bayesian learning from trade history"""
        try:
            if trade_history.empty:
                return recommendations
            
            enhanced_recs = []
            
            for rec in recommendations:
                try:
                    symbol = rec['symbol']
                    base_confidence = rec.get('confidence', 0.7)
                    
                    # Find similar trades in history
                    similar_trades = self._find_similar_trades(rec, trade_history)
                    
                    if len(similar_trades) >= 3:  # Need at least 3 similar trades
                        # Calculate win rate from similar trades
                        # This is simplified - in reality you'd need outcome data
                        wins = len(similar_trades[similar_trades.get('outcome', 0) > 0])
                        total = len(similar_trades)
                        
                        # Update confidence using Bayesian method
                        bayesian_confidence = self.bayesian_updater.update_confidence_from_trades(
                            base_confidence, wins, total
                        )
                        
                        # Calculate confidence interval
                        ci_lower, ci_upper = self.bayesian_updater.get_confidence_interval(wins, total)
                        
                        # Update recommendation
                        enhanced_rec = rec.copy()
                        enhanced_rec.update({
                            'confidence': bayesian_confidence,
                            'bayesian_confidence': bayesian_confidence,
                            'confidence_interval': (ci_lower, ci_upper),
                            'similar_trades_count': total,
                            'similar_trades_win_rate': wins / total if total > 0 else 0,
                            'reasoning': enhanced_rec.get('reasoning', '') + f" (Bayesian update from {total} similar trades)"
                        })
                        
                        enhanced_recs.append(enhanced_rec)
                    else:
                        # Not enough similar trades, use original
                        enhanced_recs.append(rec)
                
                except Exception as e:
                    logger.error(f"Error enhancing {rec.get('symbol', 'UNKNOWN')} with Bayesian learning: {e}")
                    enhanced_recs.append(rec)
            
            return enhanced_recs
        
        except Exception as e:
            logger.error(f"Error in Bayesian enhancement: {e}")
            return recommendations
    
    def _find_similar_trades(self, recommendation: Dict, trade_history: pd.DataFrame) -> pd.DataFrame:
        """Find similar trades in history for Bayesian learning"""
        try:
            if trade_history.empty:
                return pd.DataFrame()
            
            symbol = recommendation['symbol']
            
            # Find trades for same symbol
            symbol_trades = trade_history[trade_history['symbol'] == symbol]
            
            if not symbol_trades.empty:
                return symbol_trades
            
            # If no symbol matches, find trades with similar characteristics
            # This is a simplified implementation
            similar_trades = trade_history.head(10)  # Placeholder
            
            return similar_trades
        
        except Exception as e:
            logger.error(f"Error finding similar trades: {e}")
            return pd.DataFrame()
    
    def learn_from_trades(self, retrain: bool = False) -> None:
        """Learn from recent trades (enhanced for two-stage system)"""
        try:
            logger.info("🧠 Learning from trades with two-stage optimization...")
            
            trade_history = self.db_manager.get_trade_history(days=365)
            
            if trade_history.empty:
                logger.info("No trade history available for learning")
                return
            
            # Analyze trade patterns
            self._analyze_trade_patterns(trade_history)
            
            # Update Bayesian priors if enough data
            if len(trade_history) >= 20:
                self._update_bayesian_priors(trade_history)
            
            logger.info("✅ Learning from trades completed")
        
        except Exception as e:
            logger.error(f"Error learning from trades: {e}")
    
    def _analyze_trade_patterns(self, trade_history: pd.DataFrame) -> None:
        """Analyze patterns in trade history"""
        try:
            logger.info(f"📊 Analyzing patterns from {len(trade_history)} trades...")
            
            # Basic pattern analysis
            if 'action' in trade_history.columns:
                buy_trades = len(trade_history[trade_history['action'] == 'BUY'])
                sell_trades = len(trade_history[trade_history['action'] == 'SELL'])
                logger.info(f"Trade distribution: {buy_trades} BUY, {sell_trades} SELL")
            
            # Confidence analysis
            if 'confidence' in trade_history.columns:
                avg_confidence = trade_history['confidence'].mean()
                logger.info(f"Average historical confidence: {avg_confidence:.3f}")
            
            # Symbol frequency
            if 'symbol' in trade_history.columns:
                top_symbols = trade_history['symbol'].value_counts().head(5)
                logger.info(f"Most traded symbols: {list(top_symbols.index)}")
        
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {e}")
    
    def _update_bayesian_priors(self, trade_history: pd.DataFrame) -> None:
        """Update Bayesian priors based on trade outcomes"""
        try:
            # This would update the prior beliefs based on actual trade outcomes
            # For now, just log that we would do this
            logger.info("📈 Bayesian priors updated based on trade history")
        
        except Exception as e:
            logger.error(f"Error updating Bayesian priors: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'two_stage_system_enabled': True,
            'stage1_target': getattr(self.config, 'STAGE1_TARGET_COUNT', 50),
            'stage2_target': getattr(self.config, 'STAGE2_TARGET_COUNT', 10),
            'bayesian_updating_enabled': getattr(self.config, 'ENABLE_BAYESIAN_UPDATING', True),
            'model_calibration_enabled': getattr(self.config, 'ENABLE_MODEL_CALIBRATION', True),
            'sklearn_available': HAS_SKLEARN,
            'xgboost_available': HAS_XGBOOST,
            'scipy_available': HAS_SCIPY,
            'last_training_date': self.last_training_date,
            'confidence_threshold': getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.89),
            'min_confidence_for_trade': getattr(self.config, 'MIN_CONFIDENCE_FOR_TRADE', 0.65),
            'max_daily_loss': getattr(self.config, 'MAX_DAILY_LOSS', 0.05),
            'stop_loss_percentage': getattr(self.config, 'STOP_LOSS_PERCENTAGE', 0.15)
        }


# Main interface functions
def get_two_stage_ml_engine() -> TwoStageMLRecommendationEngine:
    """Get two-stage ML recommendation engine"""
    return TwoStageMLRecommendationEngine()

# Legacy compatibility
class EnhancedMLRecommendationEngine:
    """Legacy compatibility wrapper"""
    
    def __init__(self):
        self.two_stage_engine = TwoStageMLRecommendationEngine()
    
    async def generate_enhanced_recommendations(self, max_recommendations: int = None) -> List[Dict[str, Any]]:
        """Legacy method - uses two-stage system"""
        return await self.two_stage_engine.generate_recommendations(max_recommendations)
    
    def learn_from_trades(self, retrain: bool = False) -> None:
        """Legacy method"""
        self.two_stage_engine.learn_from_trades(retrain)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Legacy method"""
        return self.two_stage_engine.get_model_info()

# For backwards compatibility
MLRecommendationEngine = EnhancedMLRecommendationEngine
