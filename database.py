"""
Database management system for AI Trading System
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger
from two_stage_config import get_config

config = get_config()
Base = declarative_base()

class Portfolio(Base):
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True)
    total_value = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    invested_amount = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    num_positions = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    unrealized_pnl_percent = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_date = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    reasoning = Column(Text)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    portfolio_value_before = Column(Float)
    portfolio_value_after = Column(Float)

class AIRecommendation(Base):
    __tablename__ = 'ai_recommendations'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    price_target = Column(Float)
    stop_loss = Column(Float)
    position_size = Column(Float)
    time_horizon = Column(String(20))
    risk_level = Column(String(20))
    executed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.now)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> None:
        try:
            # Filter only valid Portfolio columns
            valid_keys = ['total_value', 'cash', 'invested_amount', 'total_return', 'daily_pnl', 'num_positions', 'timestamp']
            filtered_data = {k: v for k, v in portfolio_data.items() if k in valid_keys}

            if 'timestamp' not in filtered_data:
                filtered_data['timestamp'] = datetime.now()

            with self.get_session() as session:
                portfolio = Portfolio(**filtered_data)
                session.add(portfolio)
                session.commit()
                logger.info("Portfolio snapshot saved")
        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {e}")

    def save_trade(self, trade_data: Dict[str, Any]) -> None:
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def save_ai_recommendation(self, recommendation_data: Dict[str, Any]) -> None:
        try:
            with self.get_session() as session:
                rec = AIRecommendation(**recommendation_data)
                session.add(rec)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")

    def get_active_positions(self) -> List[Position]:
        try:
            with self.get_session() as session:
                return session.query(Position).filter_by(is_active=True).all()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                query = session.query(Portfolio).filter(Portfolio.timestamp >= cutoff_date)
                data = [
                    {'timestamp': p.timestamp, 'total_value': p.total_value, 'cash': p.cash,
                     'invested_amount': p.invested_amount, 'total_return': p.total_return,
                     'daily_pnl': p.daily_pnl, 'num_positions': p.num_positions}
                    for p in query.all()
                ]

                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                query = session.query(Trade).filter(Trade.timestamp >= cutoff_date)
                data = [
                    {'timestamp': t.timestamp, 'symbol': t.symbol, 'action': t.action,
                     'quantity': t.quantity, 'price': t.price, 'total_amount': t.total_amount,
                     'reasoning': t.reasoning, 'confidence': t.confidence}
                    for t in query.all()
                ]

                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return pd.DataFrame()

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """
        ADDED: Get all trades from the database (no date limit)
        Returns list of dictionaries instead of SQLAlchemy objects for easier processing
        """
        try:
            with self.get_session() as session:
                trades = session.query(Trade).order_by(Trade.timestamp).all()
                trade_list = []
                for t in trades:
                    trade_dict = {
                        'id': t.id,
                        'symbol': t.symbol,
                        'action': t.action,
                        'quantity': t.quantity,
                        'price': t.price,
                        'total_amount': t.total_amount,
                        'fees': t.fees,
                        'reasoning': t.reasoning,
                        'confidence': t.confidence,
                        'timestamp': t.timestamp,
                        'portfolio_value_before': t.portfolio_value_before,
                        'portfolio_value_after': t.portfolio_value_after
                    }
                    trade_list.append(trade_dict)

                logger.info(f"Retrieved {len(trade_list)} trades from database")
                return trade_list

        except Exception as e:
            logger.error(f"Error getting all trades: {e}")
            return []

    def update_trade_portfolio_value_after(self, timestamp: datetime, portfolio_value: float) -> None:
        """
        ADDED: Update the portfolio_value_after field for a specific trade
        """
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter(Trade.timestamp == timestamp).first()
                if trade:
                    trade.portfolio_value_after = portfolio_value
                    session.commit()
                    logger.debug(f"Updated portfolio value after for trade at {timestamp}")
                else:
                    logger.warning(f"Trade not found for timestamp {timestamp}")
        except Exception as e:
            logger.error(f"Error updating trade portfolio value: {e}")

    def calculate_performance_metrics(self) -> Dict[str, float]:
        return {'total_return': 0.0, 'sharpe_ratio': 0.0}

    def get_unexecuted_recommendations(self) -> List[AIRecommendation]:
        try:
            with self.get_session() as session:
                return session.query(AIRecommendation).filter_by(executed=False).all()
        except Exception as e:
            return []

db_manager = DatabaseManager()

def get_db_manager() -> DatabaseManager:
    return db_manager
