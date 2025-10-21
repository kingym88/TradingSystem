# portfolio_manager.py
"""
Portfolio Management Module for Two-Stage Trading System
UPDATED: Added live price tracking and current portfolio value calculations
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
import yfinance as yf

from database import get_db_manager
from two_stage_config import get_config

config = get_config()

class TwoStagePortfolioManager:
    """Portfolio management optimized for two-stage analysis with live price tracking"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self._price_cache = {}  # Cache for live prices
    
    def get_live_price(self, symbol: str) -> Optional[float]:
        """
        Fetch current live price from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            # Get current price from info or recent history
            try:
                current_price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
                if current_price:
                    self._price_cache[symbol] = current_price
                    return float(current_price)
            except:
                pass
            
            # Fallback: get from recent history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                self._price_cache[symbol] = current_price
                return current_price
            
            logger.warning(f"Could not fetch live price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
            return None
    
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
    
    def calculate_position_metrics(self, symbol: str, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate detailed metrics for a position including average buy price and current value
        """
        try:
            # Filter trades for this symbol
            symbol_trades = [t for t in trades if t['symbol'] == symbol]
            
            if not symbol_trades:
                return None
            
            # Calculate total shares and weighted average price
            total_shares = 0
            total_cost = 0.0
            
            for trade in symbol_trades:
                if trade['action'] == 'BUY':
                    total_shares += trade['quantity']
                    total_cost += trade['total_amount']
                elif trade['action'] == 'SELL':
                    total_shares -= trade['quantity']
                    # Reduce cost basis proportionally
                    if total_shares > 0:
                        sell_ratio = trade['quantity'] / (total_shares + trade['quantity'])
                        total_cost -= (total_cost * sell_ratio)
            
            if total_shares <= 0:
                return None
            
            avg_buy_price = total_cost / total_shares
            
            # Get current live price
            current_price = self.get_live_price(symbol)
            
            if current_price is None:
                # Fallback to last trade price
                current_price = symbol_trades[-1]['price']
                logger.warning(f"Using last trade price for {symbol}: ${current_price:.2f}")
            
            current_value = total_shares * current_price
            unrealized_pnl = current_value - total_cost
            unrealized_pnl_percent = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
            
            return {
                'symbol': symbol,
                'quantity': total_shares,
                'avg_buy_price': avg_buy_price,
                'current_price': current_price,
                'cost_basis': total_cost,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_percent': unrealized_pnl_percent
            }
            
        except Exception as e:
            logger.error(f"Error calculating position metrics for {symbol}: {e}")
            return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        UPDATED: Get current portfolio summary with live prices from Yahoo Finance
        """
        try:
            trades = self.db_manager.get_all_trades()
            
            if not trades:
                return {
                    'total_value': self.config.INITIAL_CAPITAL,
                    'cash': self.config.INITIAL_CAPITAL,
                    'invested_amount': 0.0,
                    'current_market_value': 0.0,
                    'holdings': {},
                    'holdings_details': {},
                    'total_return': 0.0,
                    'total_return_dollars': 0.0,
                    'num_positions': 0
                }
            
            # Calculate cash position from all trades
            cash = self.config.INITIAL_CAPITAL
            holdings = {}  # symbol -> quantity
            
            for trade in sorted(trades, key=lambda x: x.get('timestamp', datetime.min)):
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                total_amount = trade['total_amount']
                
                if action == 'BUY':
                    cash -= total_amount
                    holdings[symbol] = holdings.get(symbol, 0) + quantity
                elif action == 'SELL':
                    cash += total_amount
                    holdings[symbol] = holdings.get(symbol, 0) - quantity
                    if holdings[symbol] <= 0:
                        holdings.pop(symbol, None)
            
            # Calculate current market value with live prices
            holdings_details = {}
            total_market_value = 0.0
            total_cost_basis = 0.0
            
            active_symbols = [s for s, q in holdings.items() if q > 0]
            
            for symbol in active_symbols:
                position_metrics = self.calculate_position_metrics(symbol, trades)
                
                if position_metrics:
                    holdings_details[symbol] = position_metrics
                    total_market_value += position_metrics['current_value']
                    total_cost_basis += position_metrics['cost_basis']
            
            # Calculate total portfolio value
            total_value = cash + total_market_value
            
            # Calculate returns
            total_return_dollars = total_value - self.config.INITIAL_CAPITAL
            total_return_percent = (total_return_dollars / self.config.INITIAL_CAPITAL) * 100
            
            return {
                'total_value': total_value,
                'cash': cash,
                'invested_amount': total_cost_basis,
                'current_market_value': total_market_value,
                'holdings': {k: v for k, v in holdings.items() if v > 0},
                'holdings_details': holdings_details,
                'total_return': total_return_percent,
                'total_return_dollars': total_return_dollars,
                'num_positions': len(active_symbols)
            }
        
        except Exception as e:
            logger.error(f"Error calculating portfolio: {e}")
            return {
                'total_value': self.config.INITIAL_CAPITAL,
                'cash': self.config.INITIAL_CAPITAL,
                'invested_amount': 0.0,
                'current_market_value': 0.0,
                'holdings': {},
                'holdings_details': {},
                'total_return': 0.0,
                'total_return_dollars': 0.0,
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
