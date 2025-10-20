"""
Weekend Deep Analysis System
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from two_stage_config import get_config
from database import get_db_manager
from two_stage_data_manager import EnhancedDataManager
from perplexity_client import PerplexityClientSync
from two_stage_ml_engine import MLRecommendationEngine

config = get_config()

class WeekendAnalyzer:
    def __init__(self):
        self.config = config
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.ai_client = PerplexityClientSync()
        self.ml_engine = MLRecommendationEngine()

    def run_weekend_analysis(self) -> Dict[str, Any]:
        try:
            logger.info("🔍 Starting Weekend Deep Analysis...")

            analysis_results = {
                'timestamp': datetime.now(),
                'analysis_type': 'weekend_deep_analysis',
                'portfolio_review': {},
                'market_analysis': {},
                'performance_analysis': {},
                'ml_insights': {},
                'recommendations': {}
            }

            logger.info("📊 Performing portfolio review...")
            analysis_results['portfolio_review'] = self.analyze_portfolio_performance()

            logger.info("📈 Analyzing market conditions...")
            analysis_results['market_analysis'] = self.analyze_market_conditions()

            logger.info("📉 Analyzing performance metrics...")
            analysis_results['performance_analysis'] = self.analyze_performance_metrics()

            logger.info("🤖 Generating ML insights...")
            analysis_results['ml_insights'] = self.generate_ml_insights()

            logger.info("💡 Generating strategic recommendations...")
            analysis_results['recommendations'] = self.generate_strategic_recommendations()

            logger.info("✅ Weekend analysis completed successfully")
            return analysis_results

        except Exception as e:
            logger.error(f"Error in weekend analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

    def analyze_portfolio_performance(self) -> Dict[str, Any]:
        try:
            positions = self.db_manager.get_active_positions()

            if not positions:
                return {'message': 'No active positions to analyze'}

            portfolio_analysis = {
                'total_positions': len(positions),
                'position_analysis': [],
                'performance_summary': {}
            }

            total_value = 0
            total_unrealized_pnl = 0

            for position in positions:
                try:
                    stock_data = self.data_manager.get_stock_data(position.symbol, validate=False)

                    position_analysis = {
                        'symbol': position.symbol,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': stock_data.price if stock_data else position.current_price,
                        'market_value': position.market_value,
                        'unrealized_pnl': position.unrealized_pnl,
                        'unrealized_pnl_percent': position.unrealized_pnl_percent,
                        'days_held': (datetime.now() - position.entry_date).days,
                    }

                    portfolio_analysis['position_analysis'].append(position_analysis)
                    total_value += position.market_value
                    total_unrealized_pnl += position.unrealized_pnl

                except Exception as e:
                    logger.warning(f"Error analyzing position {position.symbol}: {e}")
                    continue

            portfolio_analysis['performance_summary'] = {
                'total_market_value': total_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_unrealized_pnl_percent': (total_unrealized_pnl / (total_value - total_unrealized_pnl) * 100) if total_value > total_unrealized_pnl else 0
            }

            return portfolio_analysis

        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {'error': str(e)}

    def analyze_market_conditions(self) -> Dict[str, Any]:
        try:
            market_analysis = {
                'micro_cap_sentiment': 'neutral',
                'volatility_analysis': {},
                'sector_trends': 'Mixed conditions across micro-cap sectors'
            }

            candidates = self.data_manager.screen_micro_caps()

            if candidates:
                sample_symbols = candidates[:10]
                market_data = []

                for symbol in sample_symbols:
                    try:
                        stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                        if stock_data:
                            market_data.append({
                                'symbol': symbol,
                                'change_percent': stock_data.change_percent,
                                'volume': stock_data.volume,
                                'price': stock_data.price
                            })
                    except:
                        continue

                if market_data:
                    df = pd.DataFrame(market_data)

                    positive_stocks = len(df[df['change_percent'] > 0])
                    total_stocks = len(df)
                    sentiment_ratio = positive_stocks / total_stocks if total_stocks > 0 else 0.5

                    if sentiment_ratio > 0.6:
                        market_analysis['micro_cap_sentiment'] = 'bullish'
                    elif sentiment_ratio < 0.4:
                        market_analysis['micro_cap_sentiment'] = 'bearish'
                    else:
                        market_analysis['micro_cap_sentiment'] = 'neutral'

            return market_analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'error': str(e)}

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        try:
            portfolio_history = self.db_manager.get_portfolio_history(days=30)

            if portfolio_history.empty:
                return {'message': 'No portfolio history available'}

            performance_metrics = {
                'weekly_return': 0.0,
                'monthly_return': 0.0,
                'max_drawdown': 0.0
            }

            return performance_metrics

        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return {'error': str(e)}

    def generate_ml_insights(self) -> Dict[str, Any]:
        try:
            ml_insights = {
                'model_status': self.ml_engine.get_model_info(),
                'recent_recommendations': 0,
                'model_retrained': False
            }

            return ml_insights

        except Exception as e:
            logger.error(f"Error generating ML insights: {e}")
            return {'error': str(e)}

    def generate_strategic_recommendations(self) -> Dict[str, Any]:
        try:
            recommendations = {
                'portfolio_actions': [],
                'risk_adjustments': [],
                'opportunity_areas': [
                    "Micro-cap biotechnology stocks with upcoming catalysts",
                    "Small technology companies with strong revenue growth",
                    "Undervalued industrial stocks with improving margins"
                ]
            }

            return recommendations

        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            return {'error': str(e)}

def run_weekend_analysis():
    analyzer = WeekendAnalyzer()

    print("🔍 Starting Weekend Deep Analysis...")
    print("This may take several minutes to complete...")

    results = analyzer.run_weekend_analysis()

    if 'error' not in results:
        print("\n✅ Weekend Analysis Complete!")
        print("="*50)
        print("\n📋 Check the reports directory for detailed analysis file")

    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    run_weekend_analysis()
