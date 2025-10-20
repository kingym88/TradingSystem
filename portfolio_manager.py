# portfolio_manager.py
"""
Portfolio Management Module for Two-Stage Trading System
Separated to avoid circular imports with trade_logger.py
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
from database import get_db_manager
from two_stage_config import get_config

config = get_config()


class TwoStagePortfolioManager:
    """Portfolio management optimized for two-stage analysis"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
    
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
    
    def execute_trade(self, symbol: str, action: str, quantity: int, price: float, 
                     reasoning: str = None, confidence: float = None) -> Tuple[bool, str]:
        """Execute a trade and update portfolio"""
        try:
            # Get current portfolio state
            current_portfolio = self.get_portfolio_summary()
            total_amount = quantity * price
            
            # Validate trade
            if action == 'BUY':
                if total_amount > current_portfolio['cash']:
                    return False, f"Insufficient funds: ${total_amount:.2f} > ${current_portfolio['cash']:.2f}"
            elif action == 'SELL':
                current_holdings = current_portfolio['holdings'].get(symbol, 0)
                if quantity > current_holdings:
                    return False, f"Insufficient shares: {quantity} > {current_holdings}"
            else:
                return False, f"Invalid action: {action}"
            
            # Save trade to database
            trade_data = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'total_amount': total_amount,
                'reasoning': reasoning or f"Manual {action.lower()} entry",
                'confidence': confidence,
                'timestamp': datetime.now(),
                'portfolio_value_before': current_portfolio['total_value']
            }
            
            self.db_manager.save_trade(trade_data)
            
            # Calculate new portfolio value
            new_portfolio = self.get_portfolio_summary()
            
            # Update trade with portfolio_value_after
            self.db_manager.update_trade_portfolio_value_after(
                trade_data['timestamp'], 
                new_portfolio['total_value']
            )
            
            logger.info(f"Trade executed: {action} {quantity} {symbol} @ ${price:.2f}")
            
            return True, f"Trade executed successfully: {action} {quantity} {symbol} @ ${price:.2f}"
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Trade execution error: {str(e)}"
    
    def calculate_portfolio_from_trades(self) -> Dict[str, Any]:
        """Recalculate portfolio from all trades (same as get_portfolio_summary)"""
        return self.get_portfolio_summary()


# Singleton instance
_portfolio_manager_instance = None

def get_portfolio_manager() -> TwoStagePortfolioManager:
    """Get portfolio manager singleton instance"""
    global _portfolio_manager_instance
    if _portfolio_manager_instance is None:
        _portfolio_manager_instance = TwoStagePortfolioManager()
    return _portfolio_manager_instance


# For backwards compatibility
PortfolioManager = TwoStagePortfolioManager
