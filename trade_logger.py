"""
Enhanced Trade Logging System with Bulk Import Capabilities
FEATURES:
  - CSV bulk import
  - Interactive multi-entry
  - Quick paste format
  - All original single-trade functionality
  - Portfolio validation
  - Detailed error reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
import csv
import io
import asyncio

from database import get_db_manager, Trade
from portfolio_manager import get_portfolio_manager
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


class BulkTradeLogger:
    """Enhanced Trade Logger with bulk import capabilities"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.portfolio_manager = get_portfolio_manager()
        
    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price (with fallback)"""
        try:
            from two_stage_data_manager import EnhancedDataManager
            data_manager = EnhancedDataManager()
            stock_data = data_manager.get_stock_data(symbol, validate=False)
            if stock_data:
                return stock_data.price
        except Exception as e:
            logger.warning(f"Could not fetch price for {symbol}: {e}")
        return None
    
    def validate_trade(self, trade: TradeEntry, current_portfolio: Dict) -> Tuple[bool, str]:
        """Validate if a trade can be executed"""
        try:
            total_amount = trade.quantity * trade.price
            
            if trade.action == 'BUY':
                if total_amount > current_portfolio['cash']:
                    return False, f"Insufficient funds: ${total_amount:.2f} > ${current_portfolio['cash']:.2f}"
            elif trade.action == 'SELL':
                current_holdings = current_portfolio['holdings'].get(trade.symbol, 0)
                if trade.quantity > current_holdings:
                    return False, f"Insufficient shares: {trade.quantity} > {current_holdings} for {trade.symbol}"
            else:
                return False, f"Invalid action: {trade.action}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def execute_trade(self, trade: TradeEntry) -> Tuple[bool, str]:
        """Execute a single trade"""
        try:
            success, message = self.portfolio_manager.execute_trade(
                trade.symbol,
                trade.action,
                trade.quantity,
                trade.price,
                trade.reasoning,
                trade.confidence
            )
            return success, message
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Execution error: {str(e)}"
    
    # ==================== CSV BULK IMPORT ====================
    
    def import_trades_from_csv(self, csv_file_path: str) -> Tuple[int, int, List[str]]:
        """
        Import trades from CSV file
        Returns: (success_count, failed_count, error_messages)
        """
        try:
            print("\n📂 IMPORTING TRADES FROM CSV")
            print("=" * 70)
            
            # Read CSV
            df = pd.read_csv(csv_file_path)
            
            # Validate required columns
            required_columns = ['symbol', 'action', 'quantity', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return 0, 0, [f"Missing required columns: {missing_columns}"]
            
            print(f"📊 Found {len(df)} trades in CSV file")
            print(f"Columns: {list(df.columns)}")
            
            # Preview trades
            print("\n📋 PREVIEW (first 5 trades):")
            print(df.head().to_string(index=False))
            
            confirm = input("\n⚠️  Proceed with import? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                return 0, 0, ["Import cancelled by user"]
            
            # Process trades
            trades = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    # Parse date
                    if 'date' in df.columns and pd.notna(row['date']):
                        trade_date = pd.to_datetime(row['date'])
                    else:
                        trade_date = datetime.now()
                    
                    # Parse confidence
                    confidence = None
                    if 'confidence' in df.columns and pd.notna(row['confidence']):
                        confidence = float(row['confidence'])
                        confidence = max(0.0, min(1.0, confidence))
                    
                    # Create trade entry
                    trade = TradeEntry(
                        symbol=str(row['symbol']).upper().strip(),
                        action=str(row['action']).upper().strip(),
                        quantity=int(row['quantity']),
                        price=float(row['price']),
                        date=trade_date,
                        reasoning=str(row.get('reasoning', f"CSV import - Row {idx+1}")),
                        confidence=confidence,
                        source="csv_import"
                    )
                    
                    # Validate action
                    if trade.action not in ['BUY', 'SELL']:
                        errors.append(f"Row {idx+1}: Invalid action '{trade.action}'")
                        continue
                    
                    trades.append(trade)
                    
                except Exception as e:
                    errors.append(f"Row {idx+1}: Parse error - {str(e)}")
            
            print(f"\n✅ Parsed {len(trades)} valid trades")
            if errors:
                print(f"⚠️  {len(errors)} rows had errors")
            
            # Execute trades in order
            return self._execute_bulk_trades(trades, errors)
            
        except FileNotFoundError:
            return 0, 0, [f"File not found: {csv_file_path}"]
        except Exception as e:
            logger.error(f"CSV import error: {e}")
            return 0, 0, [f"Import error: {str(e)}"]
    
    def import_trades_from_csv_content(self, csv_content: str) -> Tuple[int, int, List[str]]:
        """Import trades from CSV string content (for pasting)"""
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Validate and process similar to file import
            required_columns = ['symbol', 'action', 'quantity', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return 0, 0, [f"Missing required columns: {missing_columns}"]
            
            trades = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    trade_date = datetime.now()
                    if 'date' in df.columns and pd.notna(row['date']):
                        trade_date = pd.to_datetime(row['date'])
                    
                    confidence = None
                    if 'confidence' in df.columns and pd.notna(row['confidence']):
                        confidence = float(row['confidence'])
                    
                    trade = TradeEntry(
                        symbol=str(row['symbol']).upper().strip(),
                        action=str(row['action']).upper().strip(),
                        quantity=int(row['quantity']),
                        price=float(row['price']),
                        date=trade_date,
                        reasoning=str(row.get('reasoning', f"Bulk import - Row {idx+1}")),
                        confidence=confidence,
                        source="bulk_import"
                    )
                    
                    if trade.action in ['BUY', 'SELL']:
                        trades.append(trade)
                    else:
                        errors.append(f"Row {idx+1}: Invalid action")
                        
                except Exception as e:
                    errors.append(f"Row {idx+1}: {str(e)}")
            
            return self._execute_bulk_trades(trades, errors)
            
        except Exception as e:
            return 0, 0, [f"Parse error: {str(e)}"]
    
    # ==================== QUICK PASTE FORMAT ====================
    
    def import_trades_from_quick_format(self, quick_text: str) -> Tuple[int, int, List[str]]:
        """
        Import trades from quick format (space/tab separated)
        Format: SYMBOL ACTION QUANTITY PRICE [DATE] [REASONING]
        Example: AAPL BUY 100 150.50 2025-10-15 Strong earnings
        """
        try:
            print("\n⚡ QUICK FORMAT IMPORT")
            print("=" * 70)
            
            lines = [line.strip() for line in quick_text.strip().split('\n') if line.strip()]
            
            if not lines:
                return 0, 0, ["No trades found"]
            
            trades = []
            errors = []
            
            for idx, line in enumerate(lines, 1):
                try:
                    parts = line.split()
                    
                    if len(parts) < 4:
                        errors.append(f"Line {idx}: Insufficient data (need at least: SYMBOL ACTION QTY PRICE)")
                        continue
                    
                    symbol = parts[0].upper()
                    action = parts[1].upper()
                    quantity = int(parts[2])
                    price = float(parts[3])
                    
                    # Parse optional date
                    trade_date = datetime.now()
                    reasoning_start = 4
                    if len(parts) > 4:
                        try:
                            trade_date = datetime.strptime(parts[4], "%Y-%m-%d")
                            reasoning_start = 5
                        except ValueError:
                            pass  # Not a date, treat as reasoning
                    
                    # Parse reasoning
                    reasoning = " ".join(parts[reasoning_start:]) if len(parts) > reasoning_start else f"Quick import - Line {idx}"
                    
                    trade = TradeEntry(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=price,
                        date=trade_date,
                        reasoning=reasoning,
                        confidence=None,
                        source="quick_import"
                    )
                    
                    if action in ['BUY', 'SELL']:
                        trades.append(trade)
                    else:
                        errors.append(f"Line {idx}: Invalid action '{action}'")
                        
                except Exception as e:
                    errors.append(f"Line {idx}: Parse error - {str(e)}")
            
            print(f"\n📊 Parsed {len(trades)} trades from {len(lines)} lines")
            if errors:
                print(f"⚠️  {len(errors)} lines had errors")
            
            # Show preview
            if trades:
                print("\n📋 PREVIEW:")
                for i, trade in enumerate(trades[:5], 1):
                    print(f"  {i}. {trade.action} {trade.quantity} {trade.symbol} @ ${trade.price:.2f}")
                if len(trades) > 5:
                    print(f"  ... and {len(trades) - 5} more")
            
            confirm = input("\n⚠️  Proceed with import? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                return 0, 0, ["Import cancelled by user"]
            
            return self._execute_bulk_trades(trades, errors)
            
        except Exception as e:
            return 0, 0, [f"Quick import error: {str(e)}"]
    
    # ==================== INTERACTIVE MULTI-ENTRY ====================
    
    def log_trades_interactive_multi(self) -> Tuple[int, int]:
        """Interactive multi-trade logging session"""
        try:
            print("\n📝 INTERACTIVE MULTI-TRADE LOGGING")
            print("=" * 70)
            print("Enter trades one by one. Type 'done' to finish.")
            print()
            
            trades = []
            trade_count = 0
            
            while True:
                trade_count += 1
                print(f"\n--- Trade #{trade_count} ---")
                
                # Get portfolio status
                portfolio = self.portfolio_manager.get_portfolio_summary()
                print(f"💰 Cash Available: ${portfolio['cash']:.2f}")
                
                # Symbol
                symbol = input("Symbol (or 'done' to finish): ").strip().upper()
                if symbol.lower() == 'done':
                    break
                
                if not symbol:
                    print("❌ Symbol required")
                    trade_count -= 1
                    continue
                
                # Get current price
                current_price = self.get_stock_price(symbol)
                if current_price:
                    print(f"💹 Current price: ${current_price:.2f}")
                
                # Action
                while True:
                    action = input("Action (BUY/SELL): ").strip().upper()
                    if action in ['BUY', 'SELL']:
                        break
                    print("❌ Enter BUY or SELL")
                
                # Quantity
                while True:
                    try:
                        quantity = int(input("Quantity: ").strip())
                        if quantity > 0:
                            break
                        print("❌ Quantity must be positive")
                    except ValueError:
                        print("❌ Enter a valid number")
                
                # Price
                while True:
                    try:
                        price_input = input(f"Price (press Enter for ${current_price:.2f}): " if current_price else "Price: ").strip()
                        if not price_input and current_price:
                            price = current_price
                            break
                        price = float(price_input)
                        if price > 0:
                            break
                        print("❌ Price must be positive")
                    except ValueError:
                        print("❌ Enter a valid price")
                
                # Date
                date_input = input("Date (YYYY-MM-DD) or Enter for today: ").strip()
                trade_date = datetime.now()
                if date_input:
                    try:
                        trade_date = datetime.strptime(date_input, "%Y-%m-%d")
                    except ValueError:
                        print("⚠️  Invalid date format, using today")
                
                # Reasoning
                reasoning = input("Reasoning (optional): ").strip()
                if not reasoning:
                    reasoning = f"Interactive multi-entry #{trade_count}"
                
                # Confidence
                confidence = None
                confidence_input = input("Confidence 0-1 (optional): ").strip()
                if confidence_input:
                    try:
                        confidence = float(confidence_input)
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        pass
                
                # Create trade
                trade = TradeEntry(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    date=trade_date,
                    reasoning=reasoning,
                    confidence=confidence,
                    source="interactive_multi"
                )
                
                # Validate
                is_valid, msg = self.validate_trade(trade, portfolio)
                if not is_valid:
                    print(f"⚠️  Warning: {msg}")
                    confirm = input("Add anyway? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        trade_count -= 1
                        continue
                
                trades.append(trade)
                print(f"✅ Trade #{trade_count} added: {action} {quantity} {symbol} @ ${price:.2f}")
                
                # Continue?
                another = input("\nAdd another trade? (y/n): ").strip().lower()
                if another not in ['y', 'yes']:
                    break
            
            if not trades:
                print("\n❌ No trades to process")
                return 0, 0
            
            # Show summary and confirm
            print(f"\n📊 SUMMARY: {len(trades)} trades ready to log")
            print("-" * 70)
            for i, trade in enumerate(trades, 1):
                total = trade.quantity * trade.price
                print(f"{i}. {trade.action} {trade.quantity} {trade.symbol} @ ${trade.price:.2f} = ${total:.2f}")
            
            confirm = input("\n⚠️  Confirm and execute all trades? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Cancelled")
                return 0, 0
            
            # Execute
            success_count, failed_count, errors = self._execute_bulk_trades(trades, [])
            return success_count, failed_count
            
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            return 0, 0
        except Exception as e:
            logger.error(f"Multi-entry error: {e}")
            print(f"❌ Error: {e}")
            return 0, 0
    
    # ==================== COMMON EXECUTION ====================
    
    def _execute_bulk_trades(self, trades: List[TradeEntry], errors: List[str]) -> Tuple[int, int, List[str]]:
        """Execute a list of trades with validation and error handling"""
        try:
            print(f"\n🚀 EXECUTING {len(trades)} TRADES")
            print("=" * 70)
            
            success_count = 0
            failed_count = 0
            execution_errors = errors.copy()
            
            # Get initial portfolio
            initial_portfolio = self.portfolio_manager.get_portfolio_summary()
            print(f"💰 Starting Portfolio: ${initial_portfolio['total_value']:.2f}")
            print(f"💵 Starting Cash: ${initial_portfolio['cash']:.2f}")
            print()
            
            # Execute trades in order
            for idx, trade in enumerate(trades, 1):
                try:
                    # Get current portfolio state before each trade
                    current_portfolio = self.portfolio_manager.get_portfolio_summary()
                    
                    # Validate trade
                    is_valid, validation_msg = self.validate_trade(trade, current_portfolio)
                    
                    if not is_valid:
                        failed_count += 1
                        error_msg = f"Trade {idx} ({trade.symbol}): {validation_msg}"
                        execution_errors.append(error_msg)
                        print(f"❌ {error_msg}")
                        continue
                    
                    # Execute trade
                    success, message = self.execute_trade(trade)
                    
                    if success:
                        success_count += 1
                        total = trade.quantity * trade.price
                        print(f"✅ Trade {idx}/{len(trades)}: {trade.action} {trade.quantity} {trade.symbol} @ ${trade.price:.2f} = ${total:.2f}")
                    else:
                        failed_count += 1
                        error_msg = f"Trade {idx} ({trade.symbol}): {message}"
                        execution_errors.append(error_msg)
                        print(f"❌ {error_msg}")
                
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Trade {idx} ({trade.symbol}): Exception - {str(e)}"
                    execution_errors.append(error_msg)
                    print(f"❌ {error_msg}")
                    logger.error(f"Trade execution error: {e}")
            
            # Get final portfolio
            final_portfolio = self.portfolio_manager.get_portfolio_summary()
            
            print()
            print("=" * 70)
            print("📊 BULK IMPORT SUMMARY")
            print("=" * 70)
            print(f"✅ Successful: {success_count}")
            print(f"❌ Failed: {failed_count}")
            print(f"📋 Total: {len(trades)}")
            print()
            print(f"💰 Portfolio Change:")
            print(f"   Before: ${initial_portfolio['total_value']:.2f}")
            print(f"   After:  ${final_portfolio['total_value']:.2f}")
            print(f"   Change: ${final_portfolio['total_value'] - initial_portfolio['total_value']:+.2f}")
            print()
            print(f"💵 Cash Change:")
            print(f"   Before: ${initial_portfolio['cash']:.2f}")
            print(f"   After:  ${final_portfolio['cash']:.2f}")
            print(f"   Change: ${final_portfolio['cash'] - initial_portfolio['cash']:+.2f}")
            
            if execution_errors:
                print(f"\n⚠️  ERRORS ({len(execution_errors)}):")
                for error in execution_errors[:10]:  # Show first 10
                    print(f"   • {error}")
                if len(execution_errors) > 10:
                    print(f"   ... and {len(execution_errors) - 10} more")
            
            return success_count, failed_count, execution_errors
            
        except Exception as e:
            logger.error(f"Bulk execution error: {e}")
            return 0, len(trades), [f"Bulk execution error: {str(e)}"]
    
    # ==================== SINGLE TRADE (ORIGINAL) ====================
    
    def log_trade_interactive(self) -> bool:
        """Original single trade logging (kept for compatibility)"""
        try:
            print("\n" + "="*50)
            print("📝 SINGLE TRADE LOGGING")
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
            
            # Try to get current price
            current_price = self.get_stock_price(symbol)
            if current_price:
                print(f"✅ Current price for {symbol}: ${current_price:.2f}")
            
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
                    price_prompt = f"Price per share"
                    if current_price:
                        price_prompt += f" (current: ${current_price:.2f})"
                    price_prompt += ": "
                    
                    price_input = input(price_prompt).strip()
                    if not price_input and current_price:
                        price = current_price
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
                success, message = self.execute_trade(trade_entry)
                
                if success:
                    print(f"✅ {message}")
                    
                    # Show updated portfolio
                    updated_portfolio = self.portfolio_manager.get_portfolio_summary()
                    print(f"\n📊 PORTFOLIO UPDATE:")
                    print(f"  Previous Value: ${current_portfolio['total_value']:.2f}")
                    print(f"  Current Value: ${updated_portfolio['total_value']:.2f}")
                    print(f"  Change: ${updated_portfolio['total_value'] - current_portfolio['total_value']:+.2f}")
                    print(f"  Cash Remaining: ${updated_portfolio['cash']:.2f}")
                    print(f"  Total Return: {updated_portfolio['total_return']:.2f}%")
                    
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
    
    # ==================== EXPORT FUNCTIONALITY ====================
    
    def export_trades_to_csv(self, filename: str = None, days: int = None) -> str:
        """Export trades to CSV file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trades_export_{timestamp}.csv"
            
            # Get trades
            if days:
                trades_df = self.db_manager.get_trade_history(days=days)
            else:
                trades = self.db_manager.get_all_trades()
                trades_df = pd.DataFrame(trades)
            
            if trades_df.empty:
                print("No trades to export")
                return None
            
            # Select columns for export
            export_columns = ['timestamp', 'symbol', 'action', 'quantity', 'price', 
                            'total_amount', 'reasoning', 'confidence']
            export_df = trades_df[[col for col in export_columns if col in trades_df.columns]]
            
            # Save to CSV
            export_df.to_csv(filename, index=False)
            
            print(f"✅ Exported {len(export_df)} trades to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            print(f"❌ Export error: {e}")
            return None
    
    def create_csv_template(self, filename: str = "trade_import_template.csv") -> str:
        """Create a CSV template for bulk import"""
        try:
            template_data = {
                'symbol': ['AAPL', 'TSLA', 'GOOGL'],
                'action': ['BUY', 'BUY', 'SELL'],
                'quantity': [100, 50, 25],
                'price': [150.50, 242.30, 140.75],
                'date': ['2025-10-15', '2025-10-16', '2025-10-17'],
                'reasoning': ['Strong earnings', 'Market momentum', 'Profit taking'],
                'confidence': [0.85, 0.75, 0.80]
            }
            
            df = pd.DataFrame(template_data)
            df.to_csv(filename, index=False)
            
            print(f"✅ Created template: {filename}")
            print(f"📝 Edit this file and import using CSV bulk import option")
            return filename
            
        except Exception as e:
            logger.error(f"Template creation error: {e}")
            return None
    
    # ==================== REPORTING ====================
    
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
                    print(f"  📝 {trade['reasoning'][:60]}...")
                
                if trade.get('confidence'):
                    print(f"  🎯 Confidence: {trade['confidence']:.1%}")
                
                total_volume += abs(trade['total_amount'])
            
            print("-" * 80)
            
            print(f"\n📊 SUMMARY:")
            print(f"  Total Trades: {len(trades_df)}")
            print(f"  Buy Orders: {len(trades_df[trades_df['action'] == 'BUY'])}")
            print(f"  Sell Orders: {len(trades_df[trades_df['action'] == 'SELL'])}")
            print(f"  Total Volume: ${total_volume:.2f}")
            print(f"  Average Trade: ${total_volume / len(trades_df):.2f}")
            
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
            print(f"  Total Trades: {len(trades_df)}")
            print(f"  Buy Orders: {len(buy_trades)} (${buy_trades['total_amount'].sum():.2f})")
            print(f"  Sell Orders: {len(sell_trades)} (${sell_trades['total_amount'].sum():.2f})")
            print(f"  Net Investment: ${buy_trades['total_amount'].sum() - sell_trades['total_amount'].sum():.2f}")
            
            # Portfolio status
            print(f"\n💰 CURRENT PORTFOLIO:")
            print(f"  Total Value: ${portfolio['total_value']:.2f}")
            print(f"  Cash: ${portfolio['cash']:.2f}")
            print(f"  Invested: ${portfolio['invested_amount']:.2f}")
            print(f"  Return: {portfolio['total_return']:+.2f}%")
            print(f"  Positions: {portfolio['num_positions']}")
            
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
                    print(f"  {symbol}: {count} trades, ${volume:.2f} volume")
            
            # Current holdings
            if portfolio['holdings']:
                print(f"\n📋 CURRENT HOLDINGS:")
                for symbol, quantity in portfolio['holdings'].items():
                    try:
                        current_price = self.get_stock_price(symbol)
                        if current_price:
                            value = quantity * current_price
                            print(f"  {symbol}: {quantity:,} shares @ ${current_price:.2f} = ${value:.2f}")
                        else:
                            print(f"  {symbol}: {quantity:,} shares (price unavailable)")
                    except:
                        print(f"  {symbol}: {quantity:,} shares (price unavailable)")
            
        except Exception as e:
            logger.error(f"Error showing trade summary: {e}")
            print(f"❌ Error showing summary: {e}")


def run_enhanced_trade_logging_interface():
    """Enhanced trade logging interface with bulk import"""
    logger_instance = BulkTradeLogger()
    
    while True:
        try:
            print("\n" + "="*70)
            print("📝 ENHANCED TRADE LOGGING SYSTEM (With Bulk Import)")
            print("="*70)
            print("\n🔹 SINGLE TRADE:")
            print("  1. 📝 Log Single Trade (Original)")
            print("\n🔹 BULK IMPORT:")
            print("  2. 📂 Import from CSV File")
            print("  3. 📋 Paste CSV Data")
            print("  4. ⚡ Quick Format Import (SYMBOL ACTION QTY PRICE)")
            print("  5. 🔄 Interactive Multi-Entry")
            print("\n🔹 EXPORT:")
            print("  6. 💾 Export Trades to CSV")
            print("  7. 📄 Create CSV Template")
            print("\n🔹 VIEW:")
            print("  8. 📋 Show Recent Trades (30 days)")
            print("  9. 📊 Comprehensive Summary (90 days)")
            print(" 10. 📈 Portfolio Performance")
            print("\n 11. 🔙 Back to Main Menu")
            
            choice = input("\nSelect option (1-11): ").strip()
            
            if choice == '1':
                logger_instance.log_trade_interactive()
            
            elif choice == '2':
                csv_file = input("\nEnter CSV file path: ").strip()
                if csv_file:
                    logger_instance.import_trades_from_csv(csv_file)
            
            elif choice == '3':
                print("\n📋 PASTE CSV DATA")
                print("Format: symbol,action,quantity,price,date,reasoning,confidence")
                print("Paste your data below. When done, enter a blank line:")
                print("-" * 70)
                
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                
                if lines:
                    csv_content = "\n".join(lines)
                    logger_instance.import_trades_from_csv_content(csv_content)
            
            elif choice == '4':
                print("\n⚡ QUICK FORMAT IMPORT")
                print("Format: SYMBOL ACTION QUANTITY PRICE [DATE] [REASONING]")
                print("Example: AAPL BUY 100 150.50 2025-10-15 Strong earnings")
                print("Paste your trades below. When done, enter a blank line:")
                print("-" * 70)
                
                lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
                
                if lines:
                    quick_text = "\n".join(lines)
                    logger_instance.import_trades_from_quick_format(quick_text)
            
            elif choice == '5':
                logger_instance.log_trades_interactive_multi()
            
            elif choice == '6':
                days_input = input("\nExport trades from last N days (or Enter for all): ").strip()
                days = int(days_input) if days_input else None
                logger_instance.export_trades_to_csv(days=days)
            
            elif choice == '7':
                logger_instance.create_csv_template()
            
            elif choice == '8':
                logger_instance.show_recent_trades(30)
            
            elif choice == '9':
                logger_instance.show_trade_summary(90)
            
            elif choice == '10':
                portfolio = logger_instance.portfolio_manager.get_portfolio_summary()
                print(f"\n💰 PORTFOLIO PERFORMANCE:")
                print(f"  Initial Capital: ${logger_instance.portfolio_manager.initial_capital:.2f}")
                print(f"  Current Value: ${portfolio['total_value']:.2f}")
                print(f"  Total Return: {portfolio['total_return']:+.2f}%")
                print(f"  Cash Available: ${portfolio['cash']:.2f}")
                print(f"  Amount Invested: ${portfolio['invested_amount']:.2f}")
                print(f"  Active Positions: ${portfolio['num_positions']}")
            
            elif choice == '11':
                break
            
            else:
                print("❌ Invalid option. Please select 1-11.")
        
        except KeyboardInterrupt:
            print("\n❌ Cancelled")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error in trade logging interface: {e}")


# Keep old name for compatibility
TradeLogger = BulkTradeLogger
run_trade_logging_interface = run_enhanced_trade_logging_interface


if __name__ == "__main__":
    run_enhanced_trade_logging_interface()
