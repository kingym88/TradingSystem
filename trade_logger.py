"""
Updated Trade Logging System for ML Learning with Portfolio Tracking
UPDATED: Now properly updates portfolio values when trades are logged
"""
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

from database import get_db_manager, Trade
from two_stage_data_manager import EnhancedDataManager
from two_stage_config import get_config

config = get_config()

@dataclass
class TradeEntry:
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    date: datetime
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "manual"

class TradeLogger:
    """Updated Trade Logger with portfolio tracking"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_manager = EnhancedDataManager()
        # Import PortfolioManager locally to avoid circular import
        from two_stage_trading_system import PortfolioManager
        self.portfolio_manager = PortfolioManager()

    def log_trade_interactive(self) -> bool:
        """Interactive trade logging with portfolio update"""
        try:
            print("\n" + "="*50)
            print("📝 TRADE LOGGING SYSTEM")
            print("="*50)

            # Show current portfolio status
            current_portfolio = self.portfolio_manager.get_portfolio_summary()
            print(f"💰 Current Portfolio Value: ${current_portfolio['total_value']:.2f}")
            print(f"💵 Available Cash: ${current_portfolio['cash']:.2f}")
            print(f"📈 Total Return: {current_portfolio['total_return']:.2f}%")
            print("-" * 50)

            symbol = input("Stock Symbol (e.g., AAPL): ").strip().upper()
            if not symbol:
                print("❌ Symbol is required")
                return False

            stock_data = self.data_manager.get_stock_data(symbol, validate=False)
            if stock_data:
                print(f"✅ Current price for {symbol}: ${stock_data.price:.2f}")

            while True:
                action = input("Action (BUY/SELL): ").strip().upper()
                if action in ['BUY', 'SELL']:
                    break
                print("❌ Please enter BUY or SELL")

            while True:
                try:
                    quantity = int(input("Quantity (number of shares): ").strip())
                    if quantity > 0:
                        break
                    else:
                        print("❌ Quantity must be positive")
                except ValueError:
                    print("❌ Please enter a valid number")

            while True:
                try:
                    price_input = input(f"Price per share (current: ${stock_data.price:.2f} if available): ").strip()
                    if not price_input and stock_data:
                        price = stock_data.price
                        print(f"Using current market price: ${price:.2f}")
                        break
                    else:
                        price = float(price_input)
                        if price > 0:
                            break
                        else:
                            print("❌ Price must be positive")
                except ValueError:
                    print("❌ Please enter a valid price")

            # Validate if this trade is possible
            total_trade_amount = quantity * price
            if action == 'BUY' and total_trade_amount > current_portfolio['cash']:
                print(f"❌ Insufficient cash! Need ${total_trade_amount:.2f}, have ${current_portfolio['cash']:.2f}")
                return False

            while True:
                date_input = input("Date (YYYY-MM-DD) or press Enter for today: ").strip()
                if not date_input:
                    trade_date = datetime.now()
                    break
                else:
                    try:
                        trade_date = datetime.strptime(date_input, "%Y-%m-%d")
                        break
                    except ValueError:
                        print("❌ Please use format YYYY-MM-DD")

            reasoning = input("Reasoning (optional): ").strip()
            if not reasoning:
                reasoning = f"Manual {action.lower()} entry"

            confidence_input = input("Confidence (0-1, optional): ").strip()
            confidence = None
            if confidence_input:
                try:
                    confidence = float(confidence_input)
                    confidence = max(0, min(1, confidence))
                except ValueError:
                    confidence = None

            trade_entry = TradeEntry(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                date=trade_date,
                reasoning=reasoning,
                confidence=confidence,
                source="manual"
            )

            print("\n" + "-"*30)
            print("📋 TRADE SUMMARY:")
            print("-"*30)
            print(f"Symbol: {trade_entry.symbol}")
            print(f"Action: {trade_entry.action}")
            print(f"Quantity: {trade_entry.quantity:,}")
            print(f"Price: ${trade_entry.price:.2f}")
            print(f"Total: ${trade_entry.quantity * trade_entry.price:.2f}")
            print(f"Date: {trade_entry.date.strftime('%Y-%m-%d')}")
            print(f"Reasoning: {trade_entry.reasoning}")
            if trade_entry.confidence:
                print(f"Confidence: {trade_entry.confidence:.2f}")

            # Show expected portfolio impact
            if action == 'BUY':
                expected_cash = current_portfolio['cash'] - total_trade_amount
                print(f"\n📊 Expected Cash After Trade: ${expected_cash:.2f}")
            else:
                expected_cash = current_portfolio['cash'] + total_trade_amount
                print(f"\n📊 Expected Cash After Trade: ${expected_cash:.2f}")

            confirm = input("\nConfirm this trade? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                success, message = self.portfolio_manager.execute_trade(
                    trade_entry.symbol, 
                    trade_entry.action, 
                    trade_entry.quantity, 
                    trade_entry.price, 
                    trade_entry.reasoning, 
                    trade_entry.confidence
                )
                
                if success:
                    print(f"✅ {message}")
                    
                    # Show updated portfolio
                    updated_portfolio = self.portfolio_manager.get_portfolio_summary()
                    print(f"\n📊 PORTFOLIO UPDATE:")
                    print(f"   Previous Value: ${current_portfolio['total_value']:.2f}")
                    print(f"   Current Value:  ${updated_portfolio['total_value']:.2f}")
                    print(f"   Change:         ${updated_portfolio['total_value'] - current_portfolio['total_value']:+.2f}")
                    print(f"   Cash Remaining: ${updated_portfolio['cash']:.2f}")
                    print(f"   Total Return:   {updated_portfolio['total_return']:.2f}%")
                    
                    return True
                else:
                    print(f"❌ {message}")
                    return False
            else:
                print("❌ Trade cancelled")
                return False

        except KeyboardInterrupt:
            print("\n❌ Trade logging cancelled")
            return False
        except Exception as e:
            logger.error(f"Error in interactive trade logging: {e}")
            print(f"❌ Error: {e}")
            return False

    def log_trade(self, trade_entry: TradeEntry) -> bool:
        """Log a trade using the portfolio manager for proper tracking"""
        try:
            success, message = self.portfolio_manager.execute_trade(
                trade_entry.symbol,
                trade_entry.action,
                trade_entry.quantity,
                trade_entry.price,
                trade_entry.reasoning,
                trade_entry.confidence
            )
            
            if success:
                logger.info(f"Trade logged: {trade_entry.action} {trade_entry.quantity} {trade_entry.symbol} @ ${trade_entry.price:.2f}")
            else:
                logger.error(f"Failed to log trade: {message}")
            
            return success

        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return False

    def show_recent_trades(self, days: int = 30) -> None:
        """Show recent trades with portfolio impact"""
        try:
            trades_df = self.db_manager.get_trade_history(days=days)
            if trades_df.empty:
                print(f"📋 No trades found in the last {days} days")
                return
            
            print(f"\n📋 RECENT TRADES (Last {days} days):")
            print("=" * 80)
            
            total_volume = 0
            for _, trade in trades_df.iterrows():
                print(f"{trade['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                      f"{trade['action']} {trade['quantity']:,} {trade['symbol']} @ "
                      f"${trade['price']:.2f} = ${trade['total_amount']:.2f}")
                
                if trade.get('reasoning'):
                    print(f"   📝 {trade['reasoning'][:60]}...")
                
                if trade.get('confidence'):
                    print(f"   🎯 Confidence: {trade['confidence']:.1%}")
                
                total_volume += abs(trade['total_amount'])
                print("-" * 80)
            
            print(f"\n📊 SUMMARY:")
            print(f"   Total Trades: {len(trades_df)}")
            print(f"   Buy Orders: {len(trades_df[trades_df['action'] == 'BUY'])}")
            print(f"   Sell Orders: {len(trades_df[trades_df['action'] == 'SELL'])}")
            print(f"   Total Volume: ${total_volume:.2f}")
            print(f"   Average Trade: ${total_volume / len(trades_df):.2f}")
        
        except Exception as e:
            logger.error(f"Error showing recent trades: {e}")
            print(f"❌ Error showing trades: {e}")

    def show_trade_summary(self, days: int = 90) -> None:
        """Show comprehensive trade summary with performance metrics"""
        try:
            trades_df = self.db_manager.get_trade_history(days=days)
            portfolio = self.portfolio_manager.get_portfolio_summary()
            
            if trades_df.empty:
                print("📋 No trades found in the specified period")
                return
            
            print(f"\n📊 TRADE SUMMARY (Last {days} days):")
            print("=" * 60)
            
            # Basic stats
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            print(f"📈 TRADING ACTIVITY:")
            print(f"   Total Trades: {len(trades_df)}")
            print(f"   Buy Orders: {len(buy_trades)} (${buy_trades['total_amount'].sum():.2f})")
            print(f"   Sell Orders: {len(sell_trades)} (${sell_trades['total_amount'].sum():.2f})")
            print(f"   Net Investment: ${buy_trades['total_amount'].sum() - sell_trades['total_amount'].sum():.2f}")
            
            # Portfolio status
            print(f"\n💰 CURRENT PORTFOLIO:")
            print(f"   Total Value: ${portfolio['total_value']:.2f}")
            print(f"   Cash: ${portfolio['cash']:.2f}")
            print(f"   Invested: ${portfolio['invested_amount']:.2f}")
            print(f"   Return: {portfolio['total_return']:+.2f}%")
            print(f"   Positions: {portfolio['num_positions']}")
            
            # Most active symbols
            if not trades_df.empty:
                symbol_stats = trades_df.groupby('symbol').agg({
                    'total_amount': 'sum',
                    'symbol': 'count'
                }).rename(columns={'symbol': 'count'})
                symbol_stats = symbol_stats.sort_values('total_amount', ascending=False)
                
                print(f"\n📊 MOST ACTIVE SYMBOLS:")
                for symbol in symbol_stats.head(5).index:
                    count = symbol_stats.loc[symbol, 'count']
                    volume = symbol_stats.loc[symbol, 'total_amount']
                    print(f"   {symbol}: {count} trades, ${volume:.2f} volume")
            
            # Current holdings
            if portfolio['holdings']:
                print(f"\n📋 CURRENT HOLDINGS:")
                for symbol, quantity in portfolio['holdings'].items():
                    try:
                        stock_data = self.data_manager.get_stock_data(symbol, validate=False)
                        if stock_data:
                            value = quantity * stock_data.price
                            print(f"   {symbol}: {quantity:,} shares @ ${stock_data.price:.2f} = ${value:.2f}")
                        else:
                            print(f"   {symbol}: {quantity:,} shares (price unavailable)")
                    except:
                        print(f"   {symbol}: {quantity:,} shares (price unavailable)")
        
        except Exception as e:
            logger.error(f"Error showing trade summary: {e}")
            print(f"❌ Error showing summary: {e}")

def run_trade_logging_interface():
    """Updated trade logging interface with enhanced features"""
    trade_logger = TradeLogger()

    while True:
        try:
            print("\n" + "="*50)
            print("📝 ENHANCED TRADE LOGGING MENU")
            print("="*50)
            print("1. 📝 Log Single Trade")
            print("2. 📋 Show Recent Trades (30 days)") 
            print("3. 📊 Comprehensive Trade Summary")
            print("4. 🔄 Recalculate Portfolio from All Trades")
            print("5. 📈 Portfolio Performance")
            print("6. 🔙 Back to Main Menu")

            choice = input("\nSelect option (1-6): ").strip()

            if choice == '1':
                trade_logger.log_trade_interactive()
                
            elif choice == '2':
                trade_logger.show_recent_trades(30)
                
            elif choice == '3':
                trade_logger.show_trade_summary(90)
                
            elif choice == '4':
                print("🔄 Recalculating portfolio from all trades...")
                portfolio = trade_logger.portfolio_manager.calculate_portfolio_from_trades()
                print(f"✅ Portfolio recalculated!")
                print(f"   Total Value: ${portfolio['total_value']:.2f}")
                print(f"   Cash: ${portfolio['cash']:.2f}")
                print(f"   Invested: ${portfolio['invested_amount']:.2f}")
                print(f"   Return: {portfolio['total_return']:+.2f}%")
                
            elif choice == '5':
                portfolio = trade_logger.portfolio_manager.get_portfolio_summary()
                print(f"\n💰 PORTFOLIO PERFORMANCE:")
                print(f"   Initial Capital: ${trade_logger.portfolio_manager.initial_capital:.2f}")
                print(f"   Current Value: ${portfolio['total_value']:.2f}")
                print(f"   Total Return: {portfolio['total_return']:+.2f}%")
                print(f"   Cash Available: ${portfolio['cash']:.2f}")
                print(f"   Amount Invested: ${portfolio['invested_amount']:.2f}")
                print(f"   Active Positions: {portfolio['num_positions']}")
                
            elif choice == '6':
                break
                
            else:
                print("❌ Invalid option. Please select 1-6.")

        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in trade logging interface: {e}")

if __name__ == "__main__":
    run_trade_logging_interface()
