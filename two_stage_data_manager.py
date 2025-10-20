"""
Enhanced Data Manager with Two-Stage Filtering System
UPDATED: Implements 4000 → 50 → 10 intelligent stock selection
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from dataclasses import dataclass
from loguru import logger
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Technical indicators - Fixed import handling
HAS_PANDAS_TA = False
HAS_TALIB = False

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
    logger.info("✅ pandas-ta imported successfully")
except ImportError:
    logger.warning("⚠️  pandas-ta not available. Using basic technical indicators.")

try:
    import talib
    HAS_TALIB = True
    logger.info("✅ TA-Lib imported successfully")
except ImportError:
    logger.warning("⚠️  TA-Lib not available. Using pandas-ta and basic indicators.")

# Sentiment analysis - Fixed import handling
HAS_TRANSFORMERS = False
HAS_VADER = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
    logger.info("✅ Transformers imported successfully")
except ImportError:
    logger.warning("⚠️  Transformers not available. Advanced sentiment analysis disabled.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
    logger.info("✅ VADER Sentiment imported successfully")
except ImportError:
    logger.warning("⚠️  VADER Sentiment not available.")

try:
    from two_stage_config import get_config
    config = get_config()
except ImportError:
    logger.error("❌ Cannot import enhanced_config. Using fallback configuration.")
    # Two-stage filtering configuration
    class FallbackConfig:
        # Stage 1: Fast screening (4000 → 50)
        STAGE1_TARGET_COUNT = 50
        STAGE1_MAX_MARKET_CAP = 50_000_000_000  # $50B
        STAGE1_MIN_VOLUME = 1_000_000  # 1M volume
        STAGE1_MIN_PRICE = 2.00
        STAGE1_MAX_PRICE = 1000.00
        
        # Stage 2: Detailed analysis (50 → 10)
        STAGE2_TARGET_COUNT = 10
        STAGE2_ENABLE_SENTIMENT = True
        STAGE2_ENABLE_ADVANCED_ML = True
        
        # Technical analysis parameters
        RSI_PERIOD = 14
        RSI_OVERSOLD = 30
        RSI_OVERBOUGHT = 70
        SMA_SHORT = 20
        SMA_LONG = 50
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        ADX_PERIOD = 14
        ADX_THRESHOLD = 25.0
        
        # API keys
        ALPHA_VANTAGE_KEY = None
        NEWS_API_KEY = None
        ENABLE_NEWS_SENTIMENT = True
        MAX_NEWS_ARTICLES = 10
    
    config = FallbackConfig()

@dataclass
class StockScore:
    """Stock scoring for fast screening"""
    symbol: str
    price: float
    volume: int
    market_cap: float
    score: float
    reasons: List[str]

@dataclass
class EnhancedStockData:
    """Enhanced stock data with comprehensive metrics"""
    symbol: str
    price: float
    volume: int
    market_cap: Optional[float] = None
    change_percent: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    adx: Optional[float] = None
    
    # Alternative metrics
    beta: Optional[float] = None
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    momentum_score: Optional[float] = None
    
    # Sentiment data
    news_sentiment: Optional[float] = None
    
    # Final scoring
    final_score: Optional[float] = None

class FastScreeningEngine:
    """Stage 1: Fast screening to reduce 4000 stocks to top 50"""
    
    def __init__(self):
        self.config = config
        
    def apply_basic_filters(self, stocks: List[Dict]) -> List[Dict]:
        """Apply basic filters to eliminate obvious poor candidates"""
        logger.info(f"🔍 Stage 1a: Applying basic filters to {len(stocks)} stocks...")
        
        filtered_stocks = []
        
        for stock in stocks:
            try:
                # Basic data validation
                symbol = stock.get('symbol', '').strip()
                if not symbol or len(symbol) > 6:
                    continue
                
                name = stock.get('name', '').lower()
                
                # Exclude unwanted security types
                excluded_terms = ['warrant', 'rights', 'unit', 'preferred', 'etf', 'etn', 
                                 'note', 'bond', 'trust', 'fund']
                if any(term in name for term in excluded_terms):
                    continue
                
                # Market cap validation
                try:
                    market_cap_str = str(stock.get('marketCap', '0')).replace(',', '').replace('$', '')
                    market_cap = float(market_cap_str) if market_cap_str else 0
                except:
                    continue
                
                if market_cap <= 0 or market_cap > self.config.STAGE1_MAX_MARKET_CAP:
                    continue
                
                # Volume validation
                try:
                    volume_str = str(stock.get('volume', '0')).replace(',', '')
                    volume = int(volume_str) if volume_str else 0
                except:
                    continue
                
                if volume < self.config.STAGE1_MIN_VOLUME:
                    continue
                
                # Price validation (if available)
                try:
                    price = float(stock.get('price', 0))
                    if price > 0:
                        if price < self.config.STAGE1_MIN_PRICE or price > self.config.STAGE1_MAX_PRICE:
                            continue
                except:
                    pass  # Price might not be available in GitHub data
                
                # Industry/sector validation
                industry = stock.get('industry', '').lower()
                if any(term in industry for term in ['blank check', 'spac', 'shell']):
                    continue
                
                # Country filter
                country = stock.get('country', '')
                if country and country not in ['United States', 'US', '']:
                    continue
                
                filtered_stocks.append(stock)
                
            except Exception as e:
                logger.debug(f"Error filtering stock {stock.get('symbol', 'UNKNOWN')}: {e}")
                continue
        
        logger.info(f"✅ Stage 1a: {len(filtered_stocks)} stocks passed basic filters")
        return filtered_stocks
    
    def quick_score_stocks(self, stocks: List[Dict]) -> List[StockScore]:
        """Quickly score stocks using basic metrics"""
        logger.info(f"📊 Stage 1b: Quick scoring {len(stocks)} stocks...")
        
        scored_stocks = []
        
        # Process in batches for better performance
        batch_size = 50
        total_batches = (len(stocks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(stocks), batch_size):
            batch = stocks[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_stock = {executor.submit(self._score_single_stock, stock): stock 
                                 for stock in batch}
                
                for future in as_completed(future_to_stock):
                    stock_score = future.result()
                    if stock_score and stock_score.score > 0:
                        scored_stocks.append(stock_score)
            
            # Rate limiting
            time.sleep(0.5)
        
        logger.info(f"✅ Stage 1b: Scored {len(scored_stocks)} stocks successfully")
        return scored_stocks
    
    def _score_single_stock(self, stock: Dict) -> Optional[StockScore]:
        """Score a single stock using quick metrics"""
        try:
            symbol = stock['symbol']
            
            # Get basic price data (quick)
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            price = float(latest['Close'])
            volume = int(latest['Volume'])
            
            # Quick scoring factors
            score = 0.0
            reasons = []
            
            # Market cap scoring
            try:
                market_cap = float(str(stock.get('marketCap', '0')).replace(',', ''))
                if 1_000_000_000 <= market_cap <= 10_000_000_000:  # $1B-$10B sweet spot
                    score += 2.0
                    reasons.append("optimal_market_cap")
                elif market_cap > 0:
                    score += 1.0
            except:
                market_cap = 0
            
            # Volume scoring (higher is better for liquidity)
            if volume >= 5_000_000:  # 5M+ volume
                score += 2.0
                reasons.append("high_volume")
            elif volume >= 2_000_000:  # 2M+ volume
                score += 1.5
                reasons.append("good_volume")
            elif volume >= 1_000_000:  # 1M+ volume
                score += 1.0
                reasons.append("adequate_volume")
            
            # Price momentum (simple)
            if len(hist) >= 2:
                price_change = (price - hist.iloc[-2]['Close']) / hist.iloc[-2]['Close']
                if price_change > 0.02:  # +2% daily gain
                    score += 1.5
                    reasons.append("positive_momentum")
                elif price_change > 0:
                    score += 0.5
                    reasons.append("slight_positive")
            
            # Volume vs average (last 5 days)
            avg_volume = hist['Volume'].mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 1.5:  # 50% above average
                score += 1.0
                reasons.append("volume_surge")
            
            # Price range (avoid penny stocks and overpriced stocks)
            if 10 <= price <= 200:  # Sweet spot
                score += 1.0
                reasons.append("good_price_range")
            elif 5 <= price <= 500:
                score += 0.5
            
            # Sector preference (if available)
            sector = stock.get('sector', '').lower()
            preferred_sectors = ['technology', 'healthcare', 'consumer', 'financial']
            if any(pref in sector for pref in preferred_sectors):
                score += 0.5
                reasons.append("preferred_sector")
            
            return StockScore(
                symbol=symbol,
                price=price,
                volume=volume,
                market_cap=market_cap,
                score=score,
                reasons=reasons
            )
            
        except Exception as e:
            logger.debug(f"Error scoring stock {stock.get('symbol', 'UNKNOWN')}: {e}")
            return None
    
    def select_top_candidates(self, scored_stocks: List[StockScore]) -> List[str]:
        """Select top N candidates from scored stocks"""
        if not scored_stocks:
            return []
        
        # Sort by score (highest first)
        scored_stocks.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"🏆 Stage 1c: Top 10 scores:")
        for i, stock in enumerate(scored_stocks[:10], 1):
            reasons_str = ", ".join(stock.reasons[:3])  # Top 3 reasons
            logger.info(f"  {i}. {stock.symbol}: {stock.score:.1f} points ({reasons_str})")
        
        # Select top candidates
        top_candidates = [stock.symbol for stock in scored_stocks[:self.config.STAGE1_TARGET_COUNT]]
        
        logger.info(f"✅ Stage 1 Complete: Selected {len(top_candidates)} candidates for detailed analysis")
        return top_candidates

class DetailedAnalysisEngine:
    """Stage 2: Detailed analysis to select final 10 from top 50"""
    
    def __init__(self):
        self.config = config
        # Initialize sentiment analyzer if available
        self.sentiment_analyzer = None
        if HAS_VADER:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                pass
    
    async def analyze_top_candidates(self, symbols: List[str]) -> List[EnhancedStockData]:
        """Run detailed analysis on top candidates"""
        logger.info(f"🔬 Stage 2: Detailed analysis of {len(symbols)} top candidates...")
        
        enhanced_stocks = []
        
        # Process in smaller batches for detailed analysis
        batch_size = 10
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(symbols), batch_size):
            batch = symbols[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"🔍 Analyzing batch {batch_num}/{total_batches} ({len(batch)} stocks)")
            
            # Process batch asynchronously
            batch_tasks = [self._analyze_single_stock_detailed(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, EnhancedStockData):
                    enhanced_stocks.append(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Error analyzing {symbol}: {result}")
            
            # Rate limiting between batches
            await asyncio.sleep(1)
        
        logger.info(f"✅ Stage 2a: Completed detailed analysis of {len(enhanced_stocks)} stocks")
        return enhanced_stocks
    
    async def _analyze_single_stock_detailed(self, symbol: str) -> Optional[EnhancedStockData]:
        """Perform detailed analysis on a single stock"""
        try:
            # Get comprehensive data
            ticker = yf.Ticker(symbol)
            
            # Get info and history
            info = ticker.info or {}
            hist = ticker.history(period="6mo")
            
            if hist.empty or len(hist) < 50:
                return None
            
            # Basic stock data
            latest = hist.iloc[-1]
            price = float(latest['Close'])
            volume = int(latest['Volume'])
            
            # Calculate price change
            if len(hist) > 1:
                previous_close = hist.iloc[-2]['Close']
                change_percent = ((price - previous_close) / previous_close * 100)
            else:
                change_percent = 0.0
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(hist)
            
            # Get sentiment (if enabled)
            sentiment = None
            if self.config.STAGE2_ENABLE_SENTIMENT:
                sentiment = await self._get_basic_sentiment(symbol)
            
            # Create enhanced stock data
            enhanced_data = EnhancedStockData(
                symbol=symbol,
                price=price,
                volume=volume,
                market_cap=info.get('marketCap'),
                change_percent=change_percent,
                pe_ratio=info.get('forwardPE') or info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                beta=info.get('beta'),
                
                # Technical indicators
                rsi=technical_indicators.get('rsi'),
                macd=technical_indicators.get('macd'),
                macd_signal=technical_indicators.get('macd_signal'),
                sma_20=technical_indicators.get('sma_20'),
                sma_50=technical_indicators.get('sma_50'),
                adx=technical_indicators.get('adx'),
                volatility=technical_indicators.get('volatility'),
                
                # Calculated metrics
                momentum_score=technical_indicators.get('momentum_score', 0),
                liquidity_score=min(volume / 1_000_000, 5.0),  # Volume in millions, capped at 5
                
                # Sentiment
                news_sentiment=sentiment
            )
            
            # Calculate final comprehensive score
            enhanced_data.final_score = self._calculate_final_score(enhanced_data, technical_indicators)
            
            return enhanced_data
            
        except Exception as e:
            logger.debug(f"Error in detailed analysis of {symbol}: {e}")
            return None
    
    def _calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for detailed analysis"""
        try:
            indicators = {}
            
            # RSI
            try:
                if HAS_PANDAS_TA:
                    rsi = ta.rsi(hist_data['Close'], length=self.config.RSI_PERIOD)
                    indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else None
                else:
                    # Basic RSI calculation
                    delta = hist_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else None
            except:
                indicators['rsi'] = None
            
            # MACD
            try:
                if HAS_PANDAS_TA:
                    macd_data = ta.macd(hist_data['Close'])
                    if not macd_data.empty and len(macd_data.columns) >= 2:
                        indicators['macd'] = macd_data.iloc[-1, 0]
                        indicators['macd_signal'] = macd_data.iloc[-1, 1]
                else:
                    # Basic MACD
                    exp1 = hist_data['Close'].ewm(span=12).mean()
                    exp2 = hist_data['Close'].ewm(span=26).mean()
                    macd = exp1 - exp2
                    signal = macd.ewm(span=9).mean()
                    indicators['macd'] = macd.iloc[-1]
                    indicators['macd_signal'] = signal.iloc[-1]
            except:
                indicators['macd'] = None
                indicators['macd_signal'] = None
            
            # Moving Averages
            try:
                indicators['sma_20'] = hist_data['Close'].rolling(window=20).mean().iloc[-1]
                indicators['sma_50'] = hist_data['Close'].rolling(window=50).mean().iloc[-1]
            except:
                indicators['sma_20'] = None
                indicators['sma_50'] = None
            
            # Volatility
            try:
                returns = hist_data['Close'].pct_change().dropna()
                indicators['volatility'] = returns.std() * np.sqrt(252) if len(returns) > 1 else None
            except:
                indicators['volatility'] = None
            
            # Momentum
            try:
                current_price = hist_data['Close'].iloc[-1]
                if len(hist_data) > 20:
                    price_20d_ago = hist_data['Close'].iloc[-21]
                    momentum = (current_price - price_20d_ago) / price_20d_ago
                    indicators['momentum_score'] = momentum
                else:
                    indicators['momentum_score'] = 0
            except:
                indicators['momentum_score'] = 0
            
            # ADX (if available)
            try:
                if HAS_PANDAS_TA:
                    adx = ta.adx(hist_data['High'], hist_data['Low'], hist_data['Close'])
                    if not adx.empty:
                        adx_cols = [col for col in adx.columns if 'ADX' in col and 'DI' not in col]
                        if adx_cols:
                            indicators['adx'] = adx[adx_cols[0]].iloc[-1]
            except:
                indicators['adx'] = None
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    async def _get_basic_sentiment(self, symbol: str) -> Optional[float]:
        """Get basic sentiment analysis"""
        try:
            if not self.sentiment_analyzer:
                return None
            
            # Simple news headline analysis (placeholder)
            # In real implementation, this would call news APIs
            return None
            
        except Exception as e:
            logger.debug(f"Error getting sentiment for {symbol}: {e}")
            return None
    
    def _calculate_final_score(self, stock_data: EnhancedStockData, technical_indicators: Dict) -> float:
        """Calculate comprehensive final score"""
        try:
            score = 0.0
            
            # Technical Analysis Score (40% weight)
            technical_score = 0.0
            
            # RSI scoring
            if stock_data.rsi is not None:
                if 30 <= stock_data.rsi <= 40:  # Oversold but not extreme
                    technical_score += 2.0
                elif 40 < stock_data.rsi <= 60:  # Neutral
                    technical_score += 1.0
                elif stock_data.rsi > 70:  # Overbought
                    technical_score -= 1.0
            
            # MACD scoring
            if stock_data.macd is not None and stock_data.macd_signal is not None:
                if stock_data.macd > stock_data.macd_signal:  # Bullish
                    technical_score += 1.5
                else:
                    technical_score -= 0.5
            
            # Moving average scoring
            if stock_data.sma_20 and stock_data.sma_50 and stock_data.price:
                if stock_data.price > stock_data.sma_20 > stock_data.sma_50:  # Strong uptrend
                    technical_score += 2.0
                elif stock_data.price > stock_data.sma_20:  # Above short MA
                    technical_score += 1.0
            
            # ADX scoring (trend strength)
            if stock_data.adx is not None:
                if stock_data.adx > 25:  # Strong trend
                    technical_score += 1.0
            
            score += technical_score * 0.4
            
            # Momentum Score (25% weight)
            momentum_score = 0.0
            if stock_data.momentum_score is not None:
                if stock_data.momentum_score > 0.1:  # 10%+ momentum
                    momentum_score += 2.0
                elif stock_data.momentum_score > 0.05:  # 5%+ momentum
                    momentum_score += 1.0
                elif stock_data.momentum_score > 0:  # Positive momentum
                    momentum_score += 0.5
            
            score += momentum_score * 0.25
            
            # Liquidity Score (20% weight)
            liquidity_score = 0.0
            if stock_data.liquidity_score is not None:
                if stock_data.liquidity_score >= 3.0:  # 3M+ volume
                    liquidity_score += 2.0
                elif stock_data.liquidity_score >= 2.0:  # 2M+ volume
                    liquidity_score += 1.5
                elif stock_data.liquidity_score >= 1.0:  # 1M+ volume
                    liquidity_score += 1.0
            
            score += liquidity_score * 0.2
            
            # Volatility Score (10% weight)
            volatility_score = 0.0
            if stock_data.volatility is not None:
                if 0.15 <= stock_data.volatility <= 0.4:  # Sweet spot
                    volatility_score += 2.0
                elif 0.1 <= stock_data.volatility <= 0.6:  # Acceptable
                    volatility_score += 1.0
                elif stock_data.volatility > 0.8:  # Too volatile
                    volatility_score -= 1.0
            
            score += volatility_score * 0.1
            
            # Sentiment Score (5% weight)
            sentiment_score = 0.0
            if stock_data.news_sentiment is not None:
                sentiment_score = stock_data.news_sentiment * 2  # Scale to 0-2 range
            
            score += sentiment_score * 0.05
            
            return max(0, score)  # Ensure non-negative score
            
        except Exception as e:
            logger.error(f"Error calculating final score for {stock_data.symbol}: {e}")
            return 0.0
    
    def select_final_recommendations(self, enhanced_stocks: List[EnhancedStockData]) -> List[EnhancedStockData]:
        """Select final top 10 recommendations"""
        if not enhanced_stocks:
            return []
        
        # Sort by final score
        enhanced_stocks.sort(key=lambda x: x.final_score or 0, reverse=True)
        
        logger.info(f"🏆 Stage 2b: Top {min(10, len(enhanced_stocks))} final scores:")
        for i, stock in enumerate(enhanced_stocks[:10], 1):
            logger.info(f"  {i}. {stock.symbol}: {stock.final_score:.2f} points "
                       f"(RSI: {stock.rsi:.1f} if stock.rsi else 'N/A', "
                       f"Price: ${stock.price:.2f})")
        
        # Select final recommendations
        final_recommendations = enhanced_stocks[:self.config.STAGE2_TARGET_COUNT]
        
        logger.info(f"✅ Stage 2 Complete: Selected {len(final_recommendations)} final recommendations")
        return final_recommendations

class TwoStageDataManager:
    """Main manager implementing two-stage filtering: 4000 → 50 → 10"""
    
    def __init__(self):
        self.config = config
        self.github_manager = GitHubStockDataManager()
        self.fast_screener = FastScreeningEngine()
        self.detailed_analyzer = DetailedAnalysisEngine()
        
        logger.info("🚀 Two-Stage Data Manager initialized (4000 → 50 → 10)")
    
    async def run_two_stage_analysis(self) -> List[Dict[str, Any]]:
        """Run complete two-stage analysis"""
        try:
            start_time = time.time()
            logger.info("🎯 Starting Two-Stage Stock Analysis (4000 → 50 → 10)")
            
            # Get all stocks from GitHub
            logger.info("📡 Fetching stock universe from GitHub...")
            all_stocks = self.github_manager.fetch_all_stocks()
            
            if not all_stocks:
                logger.error("❌ No stocks retrieved from GitHub")
                return []
            
            logger.info(f"📊 Total stock universe: {len(all_stocks)} stocks")
            
            # STAGE 1: Fast screening (4000 → 50)
            logger.info("🚀 === STAGE 1: FAST SCREENING ===")
            
            # Apply basic filters
            filtered_stocks = self.fast_screener.apply_basic_filters(all_stocks)
            if not filtered_stocks:
                logger.error("❌ No stocks passed basic filters")
                return []
            
            # Quick scoring
            scored_stocks = self.fast_screener.quick_score_stocks(filtered_stocks)
            if not scored_stocks:
                logger.error("❌ No stocks could be scored")
                return []
            
            # Select top candidates
            top_candidates = self.fast_screener.select_top_candidates(scored_stocks)
            if not top_candidates:
                logger.error("❌ No top candidates selected")
                return []
            
            stage1_time = time.time() - start_time
            logger.info(f"✅ Stage 1 completed in {stage1_time:.1f} seconds")
            
            # STAGE 2: Detailed analysis (50 → 10)
            logger.info("🔬 === STAGE 2: DETAILED ANALYSIS ===")
            
            # Detailed analysis of top candidates
            enhanced_stocks = await self.detailed_analyzer.analyze_top_candidates(top_candidates)
            if not enhanced_stocks:
                logger.error("❌ No stocks completed detailed analysis")
                return []
            
            # Select final recommendations
            final_recommendations = self.detailed_analyzer.select_final_recommendations(enhanced_stocks)
            
            # Convert to recommendation format
            recommendations = []
            for stock in final_recommendations:
                # Calculate expected return based on technical analysis
                expected_return = self._estimate_expected_return(stock)
                confidence = min(0.95, max(0.65, stock.final_score / 10.0))  # Scale score to confidence
                
                recommendation = {
                    'symbol': stock.symbol,
                    'action': 'BUY',
                    'confidence': confidence,
                    'reasoning': self._generate_reasoning(stock),
                    'price_target': stock.price * (1 + expected_return),
                    'expected_return': expected_return,
                    'risk_score': 1 - confidence,
                    'current_price': stock.price,
                    'volume': stock.volume,
                    'technical_scores': {
                        'rsi': stock.rsi,
                        'macd': stock.macd,
                        'adx': stock.adx,
                        'volatility': stock.volatility
                    },
                    'sentiment_score': stock.news_sentiment or 0.0,
                    'final_score': stock.final_score
                }
                recommendations.append(recommendation)
            
            total_time = time.time() - start_time
            logger.info(f"🎯 Two-Stage Analysis Complete!")
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
            logger.info(f"📊 {len(all_stocks)} → {len(top_candidates)} → {len(final_recommendations)} stocks")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in two-stage analysis: {e}")
            return []
    
    def _estimate_expected_return(self, stock: EnhancedStockData) -> float:
        """Estimate expected return based on technical factors"""
        try:
            base_return = 0.08  # 8% base
            
            # Technical factor adjustments
            if stock.rsi and stock.rsi < 35:  # Oversold
                base_return += 0.05
            
            if stock.momentum_score and stock.momentum_score > 0.1:  # Strong momentum
                base_return += 0.04
            
            if stock.adx and stock.adx > 25:  # Strong trend
                base_return += 0.03
            
            if stock.volatility and stock.volatility > 0.3:  # High volatility premium
                base_return += 0.02
            
            return min(0.25, max(0.05, base_return))  # 5-25% range
            
        except Exception as e:
            logger.error(f"Error estimating return for {stock.symbol}: {e}")
            return 0.08
    
    def _generate_reasoning(self, stock: EnhancedStockData) -> str:
        """Generate reasoning for recommendation"""
        try:
            reasons = ["Two-stage analysis selection"]
            
            if stock.rsi and stock.rsi < 40:
                reasons.append("oversold RSI condition")
            
            if stock.momentum_score and stock.momentum_score > 0.05:
                reasons.append("positive momentum")
            
            if stock.liquidity_score and stock.liquidity_score > 2:
                reasons.append("high liquidity")
            
            if stock.adx and stock.adx > 25:
                reasons.append("strong trend")
            
            if stock.news_sentiment and stock.news_sentiment > 0.1:
                reasons.append("positive sentiment")
            
            reasoning = f"Selected from {len(reasons)} key factors: {', '.join(reasons[:4])}. "
            reasoning += f"Final score: {stock.final_score:.2f}/10.0"
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating reasoning for {stock.symbol}: {e}")
            return f"Two-stage analysis selection for {stock.symbol}"

class GitHubStockDataManager:
    """Enhanced GitHub stock data manager"""
    
    def __init__(self):
        self.github_urls = {
            'NASDAQ': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.json',
            'NYSE': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nyse/nyse_full_tickers.json',
            'AMEX': 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/amex/amex_full_tickers.json'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Two-Stage-AI-Trading-Bot/1.0'
        })
    
    def fetch_all_stocks(self) -> List[Dict]:
        """Fetch all stocks from GitHub with caching"""
        try:
            all_stocks = []
            successful_exchanges = 0
            
            for exchange in self.github_urls.keys():
                try:
                    url = self.github_urls[exchange]
                    logger.info(f"Fetching {exchange} stocks...")
                    
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    stocks = response.json()
                    for stock in stocks:
                        stock['exchange'] = exchange
                    
                    all_stocks.extend(stocks)
                    successful_exchanges += 1
                    logger.info(f"✅ {exchange}: {len(stocks)} stocks")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to fetch {exchange}: {e}")
                    continue
            
            logger.info(f"📊 Total: {len(all_stocks)} stocks from {successful_exchanges}/3 exchanges")
            return all_stocks
            
        except Exception as e:
            logger.error(f"Error fetching all stocks: {e}")
            return []

# Main interface functions
def get_two_stage_data_manager() -> TwoStageDataManager:
    """Get two-stage data manager instance"""
    return TwoStageDataManager()

async def run_two_stage_stock_analysis() -> List[Dict[str, Any]]:
    """Run two-stage stock analysis and return recommendations"""
    manager = TwoStageDataManager()
    return await manager.run_two_stage_analysis()

# Legacy compatibility functions
def get_enhanced_data_manager() -> TwoStageDataManager:
    """Legacy compatibility - returns two-stage manager"""
    return TwoStageDataManager()

class EnhancedDataManager:
    """Legacy compatibility wrapper"""
    
    def __init__(self):
        self.two_stage_manager = TwoStageDataManager()
    
    async def get_enhanced_stock_data(self, symbol: str, validate: bool = True):
        """Legacy method - runs single stock analysis"""
        try:
            analyzer = DetailedAnalysisEngine()
            return await analyzer._analyze_single_stock_detailed(symbol)
        except Exception as e:
            logger.error(f"Error in legacy enhanced stock data: {e}")
            return None
    
    def screen_enhanced_stocks(self, max_results: int = 50) -> List[str]:
        """Legacy method - runs two-stage analysis"""
        try:
            # Run synchronous version for compatibility
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            recommendations = loop.run_until_complete(self.two_stage_manager.run_two_stage_analysis())
            symbols = [rec['symbol'] for rec in recommendations]
            
            loop.close()
            return symbols[:max_results]
            
        except Exception as e:
            logger.error(f"Error in legacy screen enhanced stocks: {e}")
            return []