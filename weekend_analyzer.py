"""
Weekend Deep Analysis System for Two-Stage Trading System
UPDATED: Fully integrated with current TradingSystem components
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

# Import from current TradingSystem modules
from two_stage_config import get_config
from database import get_db_manager
from two_stage_data_manager import EnhancedDataManager
from portfolio_manager import get_portfolio_manager, TwoStagePortfolioManager
from perplexity_client import PerplexityClientSync
from two_stage_ml_engine import MLRecommendationEngine

config = get_config()


class WeekendAnalyzer:
    """Weekend deep analysis for two-stage trading system"""

    def __init__(self):
        self.config = config
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        self.portfolio_manager = get_portfolio_manager()
        self.ai_client = PerplexityClientSync()
        self.ml_engine = MLRecommendationEngine()

    def run_weekend_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive weekend analysis"""
        try:
            logger.info("🔍 Starting Weekend Deep Analysis...")

            analysis_results = {
                'timestamp': datetime.now(),
                'analysis_type': 'weekend_deep_analysis',
                'portfolio_review': {},
                'market_analysis': {},
                'performance_analysis': {},
                'ml_insights': {},
                'recommendations': {},
                'risk_assessment': {}
            }

            # Portfolio Review with live prices
            logger.info("📊 Performing portfolio review...")
            analysis_results['portfolio_review'] = self.analyze_portfolio_performance()

            # Market Conditions Analysis
            logger.info("📈 Analyzing market conditions...")
            analysis_results['market_analysis'] = self.analyze_market_conditions()

            # Performance Metrics
            logger.info("📉 Analyzing performance metrics...")
            analysis_results['performance_analysis'] = self.analyze_performance_metrics()

            # ML Insights
            logger.info("🤖 Generating ML insights...")
            analysis_results['ml_insights'] = self.generate_ml_insights()

            # Risk Assessment
            logger.info("⚠️ Performing risk assessment...")
            analysis_results['risk_assessment'] = self.assess_portfolio_risk()

            # Strategic Recommendations
            logger.info("💡 Generating strategic recommendations...")
            analysis_results['recommendations'] = self.generate_strategic_recommendations(
                analysis_results
            )

            # Save analysis report
            self.save_analysis_report(analysis_results)

            logger.info("✅ Weekend analysis completed successfully")
            return analysis_results

        except Exception as e:
            logger.error(f"Error in weekend analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

    def analyze_portfolio_performance(self) -> Dict[str, Any]:
        """Analyze current portfolio using portfolio_manager with live prices"""
        try:
            # Get current portfolio summary with live prices
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()

            if portfolio_summary['num_positions'] == 0:
                return {
                    'message': 'No active positions to analyze',
                    'total_value': portfolio_summary['total_value'],
                    'cash': portfolio_summary['cash']
                }

            # Enhanced portfolio analysis
            portfolio_analysis = {
                'summary': {
                    'total_value': portfolio_summary['total_value'],
                    'cash': portfolio_summary['cash'],
                    'invested_amount': portfolio_summary['invested_amount'],
                    'current_market_value': portfolio_summary['current_market_value'],
                    'total_return_dollars': portfolio_summary['total_return_dollars'],
                    'total_return_percent': portfolio_summary['total_return'],
                    'num_positions': portfolio_summary['num_positions']
                },
                'positions': [],
                'winners': [],
                'losers': [],
                'performance_stats': {}
            }

            # Analyze individual positions
            for symbol, details in portfolio_summary['holdings_details'].items():
                position_data = {
                    'symbol': symbol,
                    'quantity': details['quantity'],
                    'avg_buy_price': details['avg_buy_price'],
                    'current_price': details['current_price'],
                    'cost_basis': details['cost_basis'],
                    'current_value': details['current_value'],
                    'unrealized_pnl': details['unrealized_pnl'],
                    'unrealized_pnl_percent': details['unrealized_pnl_percent']
                }

                portfolio_analysis['positions'].append(position_data)

                # Categorize as winner or loser
                if details['unrealized_pnl'] > 0:
                    portfolio_analysis['winners'].append(position_data)
                else:
                    portfolio_analysis['losers'].append(position_data)

            # Calculate performance statistics
            if portfolio_analysis['positions']:
                pnl_values = [p['unrealized_pnl_percent'] for p in portfolio_analysis['positions']]
                portfolio_analysis['performance_stats'] = {
                    'avg_position_return': np.mean(pnl_values),
                    'best_performer': max(pnl_values),
                    'worst_performer': min(pnl_values),
                    'winners_count': len(portfolio_analysis['winners']),
                    'losers_count': len(portfolio_analysis['losers']),
                    'win_rate': len(portfolio_analysis['winners']) / len(portfolio_analysis['positions']) * 100
                }

            return portfolio_analysis

        except Exception as e:
            logger.error(f"Error analyzing portfolio performance: {e}")
            return {'error': str(e)}

    def analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions for micro-caps"""
        try:
            market_analysis = {
                'micro_cap_sentiment': 'neutral',
                'volatility_analysis': {},
                'sector_trends': {},
                'market_breadth': {}
            }

            # Screen for current micro-cap candidates
            logger.info("Screening micro-cap market...")
            stage1_candidates = self.data_manager.get_stage1_candidates()

            if stage1_candidates and len(stage1_candidates) > 0:
                # Sample top candidates for market sentiment
                sample_size = min(20, len(stage1_candidates))
                sample_symbols = stage1_candidates[:sample_size]

                market_data = []
                for symbol in sample_symbols:
                    try:
                        stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                        if stock_data and hasattr(stock_data, 'price'):
                            market_data.append({
                                'symbol': symbol,
                                'price': stock_data.price,
                                'change_percent': getattr(stock_data, 'change_percent', 0),
                                'volume': getattr(stock_data, 'volume', 0),
                                'market_cap': getattr(stock_data, 'market_cap', 0)
                            })
                    except Exception as e:
                        logger.debug(f"Error fetching data for {symbol}: {e}")
                        continue

                if market_data:
                    df = pd.DataFrame(market_data)

                    # Calculate market sentiment
                    positive_stocks = len(df[df['change_percent'] > 0])
                    total_stocks = len(df)
                    sentiment_ratio = positive_stocks / total_stocks if total_stocks > 0 else 0.5

                    if sentiment_ratio > 0.6:
                        market_analysis['micro_cap_sentiment'] = 'bullish'
                    elif sentiment_ratio < 0.4:
                        market_analysis['micro_cap_sentiment'] = 'bearish'
                    else:
                        market_analysis['micro_cap_sentiment'] = 'neutral'

                    # Market breadth metrics
                    market_analysis['market_breadth'] = {
                        'stocks_analyzed': total_stocks,
                        'advancing': positive_stocks,
                        'declining': total_stocks - positive_stocks,
                        'advance_decline_ratio': sentiment_ratio,
                        'avg_change_percent': df['change_percent'].mean(),
                        'avg_volume': df['volume'].mean()
                    }

                    # Volatility analysis
                    market_analysis['volatility_analysis'] = {
                        'price_volatility': df['change_percent'].std(),
                        'volume_volatility': df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 0
                    }

            return market_analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'error': str(e)}

    def analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze historical performance metrics"""
        try:
            # Get trade history
            all_trades = self.db_manager.get_all_trades()

            if not all_trades:
                return {'message': 'No trade history available'}

            performance_metrics = {
                'total_trades': len(all_trades),
                'trade_breakdown': {},
                'recent_performance': {},
                'trade_statistics': {}
            }

            # Analyze trade breakdown
            buy_trades = [t for t in all_trades if t.get('action') == 'BUY']
            sell_trades = [t for t in all_trades if t.get('action') == 'SELL']

            performance_metrics['trade_breakdown'] = {
                'total_buys': len(buy_trades),
                'total_sells': len(sell_trades),
                'total_buy_amount': sum(t.get('total_amount', 0) for t in buy_trades),
                'total_sell_amount': sum(t.get('total_amount', 0) for t in sell_trades)
            }

            # Calculate realized gains/losses from sells
            if sell_trades:
                realized_pnl = performance_metrics['trade_breakdown']['total_sell_amount'] - \
                              performance_metrics['trade_breakdown']['total_buy_amount']
                performance_metrics['trade_breakdown']['realized_pnl'] = realized_pnl

            # Recent performance (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_trades = [t for t in all_trades 
                           if t.get('timestamp', datetime.min) > thirty_days_ago]

            if recent_trades:
                performance_metrics['recent_performance'] = {
                    'trades_last_30_days': len(recent_trades),
                    'most_active_symbols': self._get_most_active_symbols(recent_trades)
                }

            return performance_metrics

        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return {'error': str(e)}

    def _get_most_active_symbols(self, trades: List[Dict]) -> List[Dict]:
        """Get most actively traded symbols"""
        symbol_counts = {}
        for trade in trades:
            symbol = trade.get('symbol')
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'symbol': s, 'trade_count': c} for s, c in sorted_symbols[:5]]

    def generate_ml_insights(self) -> Dict[str, Any]:
        """Generate insights from ML engine"""
        try:
            ml_insights = {
                'model_status': {},
                'recent_recommendations': 0,
                'model_performance': {},
                'confidence_metrics': {}
            }

            # Get ML model information
            try:
                model_info = self.ml_engine.get_model_info()
                ml_insights['model_status'] = model_info
            except Exception as e:
                logger.warning(f"Could not retrieve ML model info: {e}")
                ml_insights['model_status'] = {'status': 'Not initialized'}

            # Get recent recommendations from database
            try:
                recent_recommendations = self.db_manager.get_recent_recommendations(days=7)
                ml_insights['recent_recommendations'] = len(recent_recommendations) if recent_recommendations else 0

                if recent_recommendations:
                    confidences = [r.get('confidence', 0) for r in recent_recommendations]
                    ml_insights['confidence_metrics'] = {
                        'avg_confidence': np.mean(confidences),
                        'max_confidence': max(confidences),
                        'min_confidence': min(confidences)
                    }
            except Exception as e:
                logger.warning(f"Could not retrieve recent recommendations: {e}")

            return ml_insights

        except Exception as e:
            logger.error(f"Error generating ML insights: {e}")
            return {'error': str(e)}

    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess current portfolio risk metrics"""
        try:
            portfolio = self.portfolio_manager.get_portfolio_summary()

            risk_assessment = {
                'concentration_risk': {},
                'capital_allocation': {},
                'risk_metrics': {}
            }

            # Concentration risk
            if portfolio['num_positions'] > 0:
                positions = portfolio['holdings_details']
                total_value = portfolio['current_market_value']

                # Calculate position concentrations
                position_weights = {}
                for symbol, details in positions.items():
                    weight = (details['current_value'] / total_value * 100) if total_value > 0 else 0
                    position_weights[symbol] = weight

                max_concentration = max(position_weights.values()) if position_weights else 0

                risk_assessment['concentration_risk'] = {
                    'max_position_weight': max_concentration,
                    'num_positions': portfolio['num_positions'],
                    'diversification_score': min(100, portfolio['num_positions'] * 10),  # Simple score
                    'position_weights': position_weights
                }

            # Capital allocation
            risk_assessment['capital_allocation'] = {
                'cash_percent': (portfolio['cash'] / portfolio['total_value'] * 100) if portfolio['total_value'] > 0 else 100,
                'invested_percent': (portfolio['current_market_value'] / portfolio['total_value'] * 100) if portfolio['total_value'] > 0 else 0,
                'total_value': portfolio['total_value']
            }

            # Risk metrics
            risk_assessment['risk_metrics'] = {
                'max_position_size_limit': self.config.MAX_POSITION_SIZE * 100,
                'stop_loss_percentage': self.config.STOP_LOSS_PERCENTAGE * 100,
                'take_profit_percentage': self.config.TAKE_PROFIT_PERCENTAGE * 100,
                'max_positions_limit': self.config.MAX_POSITIONS
            }

            return risk_assessment

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {'error': str(e)}

    def generate_strategic_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic recommendations based on all analysis"""
        try:
            recommendations = {
                'portfolio_actions': [],
                'risk_adjustments': [],
                'opportunity_areas': [],
                'ai_insights': None
            }

            # Portfolio-based recommendations
            portfolio_review = analysis_results.get('portfolio_review', {})
            if 'performance_stats' in portfolio_review:
                stats = portfolio_review['performance_stats']

                # Suggest actions for losing positions
                if portfolio_review.get('losers'):
                    for loser in portfolio_review['losers']:
                        if loser['unrealized_pnl_percent'] < -self.config.STOP_LOSS_PERCENTAGE * 100:
                            recommendations['portfolio_actions'].append({
                                'action': 'CONSIDER_SELL',
                                'symbol': loser['symbol'],
                                'reason': f"Position down {loser['unrealized_pnl_percent']:.1f}%, approaching stop-loss",
                                'priority': 'HIGH'
                            })

                # Suggest profit-taking for winners
                if portfolio_review.get('winners'):
                    for winner in portfolio_review['winners']:
                        if winner['unrealized_pnl_percent'] > self.config.TAKE_PROFIT_PERCENTAGE * 100:
                            recommendations['portfolio_actions'].append({
                                'action': 'CONSIDER_TAKE_PROFIT',
                                'symbol': winner['symbol'],
                                'reason': f"Position up {winner['unrealized_pnl_percent']:.1f}%, at take-profit target",
                                'priority': 'MEDIUM'
                            })

            # Risk-based recommendations
            risk_assessment = analysis_results.get('risk_assessment', {})
            if 'concentration_risk' in risk_assessment:
                conc = risk_assessment['concentration_risk']
                if conc.get('max_position_weight', 0) > self.config.MAX_POSITION_SIZE * 100:
                    recommendations['risk_adjustments'].append({
                        'issue': 'CONCENTRATION_RISK',
                        'description': f"Largest position is {conc['max_position_weight']:.1f}% of portfolio",
                        'recommendation': 'Consider rebalancing to reduce concentration risk',
                        'priority': 'MEDIUM'
                    })

            # Market-based opportunities
            market_analysis = analysis_results.get('market_analysis', {})
            sentiment = market_analysis.get('micro_cap_sentiment', 'neutral')

            if sentiment == 'bullish':
                recommendations['opportunity_areas'].append(
                    "Market sentiment is bullish - consider increasing exposure to high-confidence Stage 2 selections"
                )
            elif sentiment == 'bearish':
                recommendations['opportunity_areas'].append(
                    "Market sentiment is bearish - maintain higher cash reserves and focus on quality selections"
                )
            else:
                recommendations['opportunity_areas'].append(
                    "Market sentiment is neutral - maintain balanced approach with selective new positions"
                )

            # Add general opportunity areas based on two-stage system
            recommendations['opportunity_areas'].extend([
                "Focus on Stage 2 stocks with confidence > 0.85 for new positions",
                "Monitor stocks with strong technical indicators (RSI, MACD alignment)",
                "Look for volume surges in Stage 1 candidates for early entry opportunities"
            ])

            # Get AI insights if available
            try:
                if hasattr(self.ai_client, 'get_market_insights'):
                    ai_insights = self.ai_client.get_market_insights()
                    recommendations['ai_insights'] = ai_insights
            except Exception as e:
                logger.debug(f"AI insights not available: {e}")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            return {'error': str(e)}

    def save_analysis_report(self, analysis_results: Dict[str, Any]) -> None:
        """Save analysis report to file"""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = self.config.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = reports_dir / f"weekend_analysis_{timestamp}.txt"

            # Format and write report
            with open(filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Weekend Analysis Report - {analysis_results['timestamp']}\n")
                f.write("=" * 80 + "\n\n")

                # Portfolio Summary
                f.write("PORTFOLIO SUMMARY\n")
                f.write("-" * 80 + "\n")
                portfolio = analysis_results.get('portfolio_review', {})
                if 'summary' in portfolio:
                    summary = portfolio['summary']
                    f.write(f"Total Value: ${summary['total_value']:.2f}\n")
                    f.write(f"Cash: ${summary['cash']:.2f}\n")
                    f.write(f"Invested: ${summary['invested_amount']:.2f}\n")
                    f.write(f"Market Value: ${summary['current_market_value']:.2f}\n")
                    f.write(f"Total Return: {summary['total_return_percent']:.2f}% (${summary['total_return_dollars']:.2f})\n")
                    f.write(f"Positions: {summary['num_positions']}\n")
                f.write("\n")

                # Market Analysis
                f.write("MARKET ANALYSIS\n")
                f.write("-" * 80 + "\n")
                market = analysis_results.get('market_analysis', {})
                f.write(f"Sentiment: {market.get('micro_cap_sentiment', 'N/A')}\n")
                if 'market_breadth' in market:
                    breadth = market['market_breadth']
                    f.write(f"Stocks Analyzed: {breadth.get('stocks_analyzed', 0)}\n")
                    f.write(f"Advancing: {breadth.get('advancing', 0)}\n")
                    f.write(f"Declining: {breadth.get('declining', 0)}\n")
                f.write("\n")

                # Recommendations
                f.write("STRATEGIC RECOMMENDATIONS\n")
                f.write("-" * 80 + "\n")
                recommendations = analysis_results.get('recommendations', {})

                if recommendations.get('portfolio_actions'):
                    f.write("\nPortfolio Actions:\n")
                    for action in recommendations['portfolio_actions']:
                        f.write(f"  - [{action['priority']}] {action['action']}: {action['symbol']}\n")
                        f.write(f"    Reason: {action['reason']}\n")

                if recommendations.get('opportunity_areas'):
                    f.write("\nOpportunity Areas:\n")
                    for opp in recommendations['opportunity_areas']:
                        f.write(f"  - {opp}\n")

                f.write("\n" + "=" * 80 + "\n")

            logger.info(f"Analysis report saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving analysis report: {e}")


def run_weekend_analysis():
    """Run weekend analysis from command line"""
    analyzer = WeekendAnalyzer()

    print("🔍 Starting Weekend Deep Analysis...")
    print("This may take several minutes to complete...")
    print()

    results = analyzer.run_weekend_analysis()

    if 'error' not in results:
        print("\n✅ Weekend Analysis Complete!")
        print("=" * 50)
        print()

        # Display key results
        if 'portfolio_review' in results and 'summary' in results['portfolio_review']:
            summary = results['portfolio_review']['summary']
            print("📊 Portfolio Summary:")
            print(f"   Total Value: ${summary['total_value']:,.2f}")
            print(f"   Total Return: {summary['total_return_percent']:.2f}%")
            print(f"   Positions: {summary['num_positions']}")
            print()

        if 'market_analysis' in results:
            market = results['market_analysis']
            print(f"📈 Market Sentiment: {market.get('micro_cap_sentiment', 'N/A')}")
            print()

        if 'recommendations' in results:
            recs = results['recommendations']
            if recs.get('portfolio_actions'):
                print(f"💡 Portfolio Actions: {len(recs['portfolio_actions'])} recommended")
            print()

        print("📋 Check the reports directory for detailed analysis file")
        print()
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    run_weekend_analysis()
