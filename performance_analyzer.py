"""
Performance Analysis Tool for Enhanced AI Trading System
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from database import get_db_manager
from two_stage_config import get_config

config = get_config()

class PerformanceAnalyzer:
    def __init__(self):
        self.db_manager = get_db_manager()

    def generate_comprehensive_report(self, days: int = 90) -> Dict[str, Any]:
        print("Generating comprehensive performance report...")

        portfolio_history = self.db_manager.get_portfolio_history(days)
        trade_history = self.db_manager.get_trade_history(days)

        if portfolio_history.empty:
            print("No portfolio data available for analysis")
            return {}

        returns = self._calculate_returns(portfolio_history)
        trade_metrics = self._calculate_trade_metrics(trade_history)

        report = {
            'analysis_period': f"{days} days",
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'returns': returns,
            'trade_metrics': trade_metrics,
            'portfolio_data': portfolio_history.to_dict('records'),
            'trade_data': trade_history.to_dict('records')
        }

        return report

    def _calculate_returns(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        if portfolio_df.empty or len(portfolio_df) < 2:
            return {}

        portfolio_df = portfolio_df.sort_values('timestamp')
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]

        total_return = ((final_value - initial_value) / initial_value) * 100

        return {
            'total_return': round(total_return, 2),
            'initial_value': round(initial_value, 2),
            'final_value': round(final_value, 2)
        }

    def _calculate_trade_metrics(self, trade_df: pd.DataFrame) -> Dict[str, Any]:
        if trade_df.empty:
            return {'total_trades': 0}

        total_trades = len(trade_df)
        buy_trades = len(trade_df[trade_df['action'] == 'BUY'])
        sell_trades = len(trade_df[trade_df['action'] == 'SELL'])

        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'unique_symbols': trade_df['symbol'].nunique()
        }

    def print_summary_report(self, days: int = 30):
        report = self.generate_comprehensive_report(days)

        if not report:
            print("No data available for analysis")
            return

        print("\n" + "="*60)
        print("🚀 ENHANCED AI TRADING SYSTEM - PERFORMANCE REPORT")
        print("="*60)
        print(f"📅 Analysis Period: {report['analysis_period']}")
        print(f"🕐 Generated: {report['report_date']}")
        print()

        print("📈 RETURNS ANALYSIS")
        print("-" * 30)
        returns = report.get('returns', {})
        for key, value in returns.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:.<25} {value:>10}")

        print("\n📊 TRADING METRICS")
        print("-" * 30)
        trades = report.get('trade_metrics', {})
        for key, value in trades.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key:.<25} {value:>10}")

        print("\n" + "="*60)

def main():
    analyzer = PerformanceAnalyzer()

    print("Enhanced AI Trading System - Performance Analyzer")
    print("=" * 50)

    try:
        days = input("Enter analysis period in days (default 30): ").strip()
        days = int(days) if days else 30
    except ValueError:
        days = 30
        print("Using default 30 days")

    analyzer.print_summary_report(days)

if __name__ == "__main__":
    main()
