"""
Perplexity API Client for AI Trading System
"""
import httpx
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from loguru import logger
from two_stage_config import get_config

config = get_config()

@dataclass
class TradingRecommendation:
    action: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    reasoning: str
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    time_horizon: Optional[str] = None
    risk_level: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerplexityClient:
    def __init__(self):
        self.api_key = config.PERPLEXITY_API_KEY
        self.base_url = config.PERPLEXITY_BASE_URL
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_trading_recommendation(self, portfolio_data: Dict[str, Any],
                                        market_data: Dict[str, Any],
                                        research_context: Optional[str] = None) -> List[TradingRecommendation]:
        try:
            prompt = self._build_trading_prompt(portfolio_data, market_data, research_context)
            response = await self._make_api_request(prompt)
            recommendations = self._parse_recommendations(response)
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trading recommendation: {e}")
            return []

    def _build_trading_prompt(self, portfolio_data: Dict[str, Any], 
                            market_data: Dict[str, Any], research_context: Optional[str] = None) -> str:
        prompt = f"""
        You are a professional portfolio strategist managing a micro-cap stock portfolio.

        CURRENT PORTFOLIO STATUS:
        - Total Value: ${portfolio_data.get('total_value', 0):.2f}
        - Cash Available: ${portfolio_data.get('cash', 0):.2f}
        - Number of Positions: {portfolio_data.get('num_positions', 0)}
        - Current Holdings: {portfolio_data.get('holdings', {})}
        - Total Return: {portfolio_data.get('total_return', 0):.2f}%

        MARKET CONDITIONS:
        - Market Status: {'Open' if market_data.get('market_open', False) else 'Closed'}
        - Market Sentiment: {market_data.get('sentiment', 'Neutral')}

        INVESTMENT CONSTRAINTS:
        - Only US-listed micro-cap stocks (market cap < $300M)
        - Maximum position size: 30% of portfolio
        - Stop-loss at 15% loss, take profit at 25% gain
        - Minimum daily volume: 100,000 shares

        Provide 1-3 specific trading recommendations in JSON format:
        {{"action": "BUY|SELL|HOLD", "symbol": "SYMBOL", "reasoning": "explanation", "confidence": 0.85}}

        Focus on stocks with strong fundamentals and upcoming catalysts.
        """
        return prompt.strip()

    async def _make_api_request(self, prompt: str) -> str:
        try:
            payload = {
                "model": "llama-3.1-sonar-large-128k-online",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = await self.client.post(f"{self.base_url}/chat/completions",
                                            headers=self.headers, json=payload)

            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code}")
                return ""

            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return ""

    def _parse_recommendations(self, response: str) -> List[TradingRecommendation]:
        recommendations = []
        try:
            import re
            json_pattern = r'\{[^{}]*"action"[^{}]*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    data = json.loads(match)
                    recommendation = TradingRecommendation(
                        action=data.get("action", "HOLD").upper(),
                        symbol=data.get("symbol", "").upper(),
                        reasoning=data.get("reasoning", ""),
                        confidence=float(data.get("confidence", 0.5)),
                        price_target=data.get("price_target"),
                        stop_loss=data.get("stop_loss"),
                        position_size=data.get("position_size"),
                        time_horizon=data.get("time_horizon"),
                        risk_level=data.get("risk_level")
                    )
                    recommendations.append(recommendation)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        except Exception as e:
            logger.error(f"Error parsing recommendations: {e}")

        return recommendations

    async def get_market_research(self, symbols: List[str]) -> str:
        try:
            symbols_str = ", ".join(symbols)
            prompt = f"Provide research analysis for these micro-cap stocks: {symbols_str}"
            response = await self._make_api_request(prompt)
            return response
        except Exception as e:
            logger.error(f"Error getting market research: {e}")
            return ""

    async def close(self):
        await self.client.aclose()

class PerplexityClientSync:
    def __init__(self):
        self.async_client = PerplexityClient()

    def get_trading_recommendation(self, portfolio_data: Dict[str, Any], market_data: Dict[str, Any],
                                 research_context: Optional[str] = None) -> List[TradingRecommendation]:
        return asyncio.run(self.async_client.get_trading_recommendation(
            portfolio_data, market_data, research_context))

    def get_market_research(self, symbols: List[str]) -> str:
        return asyncio.run(self.async_client.get_market_research(symbols))
