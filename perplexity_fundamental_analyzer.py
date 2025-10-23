"""
Perplexity AI Fundamental Analyzer for Trading System
Uses sonar-deep-research to analyze fundamental strength of stocks
"""
import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Perplexity AI SDK
try:
    from perplexity import Perplexity
    HAS_PERPLEXITY = True
    logger.info("✅ Perplexity AI SDK imported successfully")
except ImportError:
    HAS_PERPLEXITY = False
    logger.warning("⚠️ Perplexity AI SDK not available. Install with: pip install perplexityai")

try:
    from two_stage_config import get_config
    config = get_config()
except ImportError:
    class FallbackConfig:
        PERPLEXITY_API_KEY = None
        ENABLE_FUNDAMENTAL_ANALYSIS = True
        FUNDAMENTAL_ANALYSIS_MODEL = "sonar-deep-research"
        FUNDAMENTAL_ANALYSIS_TIMEOUT = 120  # 2 minutes per stock
    config = FallbackConfig()


@dataclass
class FundamentalAnalysis:
    """Fundamental analysis results for a stock"""
    symbol: str
    analysis_date: datetime

    # Financial Health Metrics
    pe_ratio_assessment: Optional[str] = None
    pb_ratio_assessment: Optional[str] = None
    roe_assessment: Optional[str] = None
    debt_level_assessment: Optional[str] = None
    financial_health_score: float = 0.0

    # Growth Metrics
    revenue_growth_trend: Optional[str] = None
    earnings_growth_trend: Optional[str] = None
    growth_sustainability: Optional[str] = None
    growth_score: float = 0.0

    # Competitive Position
    competitive_moat: Optional[str] = None
    market_position: Optional[str] = None
    competitive_advantages: List[str] = None
    competitive_score: float = 0.0

    # Management & Governance
    management_quality: Optional[str] = None
    governance_assessment: Optional[str] = None
    insider_activity: Optional[str] = None
    management_score: float = 0.0

    # Industry Outlook
    industry_trends: Optional[str] = None
    sector_outlook: Optional[str] = None
    regulatory_environment: Optional[str] = None
    industry_score: float = 0.0

    # Overall Assessment
    overall_score: float = 0.0
    recommendation: Optional[str] = None
    key_insights: List[str] = None
    risks: List[str] = None
    catalysts: List[str] = None

    # Raw analysis
    full_analysis_text: Optional[str] = None


class PerplexityFundamentalAnalyzer:
    """
    Uses Perplexity AI sonar-deep-research to perform comprehensive fundamental analysis
    """

    def __init__(self):
        self.config = config
        self.api_key = os.getenv('PERPLEXITY_API_KEY') or getattr(config, 'PERPLEXITY_API_KEY', None)
        self.model = getattr(config, 'FUNDAMENTAL_ANALYSIS_MODEL', 'sonar-deep-research')
        self.client = None

        if HAS_PERPLEXITY and self.api_key:
            try:
                self.client = Perplexity(api_key=self.api_key)
                logger.info("✅ Perplexity AI client initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Perplexity client: {e}")
                self.client = None
        else:
            if not HAS_PERPLEXITY:
                logger.warning("⚠️ Perplexity SDK not installed")
            if not self.api_key:
                logger.warning("⚠️ PERPLEXITY_API_KEY not set in environment")

    async def analyze_stock_fundamentals(self, symbol: str, company_name: str = None) -> Optional[FundamentalAnalysis]:
        """
        Perform comprehensive fundamental analysis on a stock using Perplexity AI

        Args:
            symbol: Stock ticker symbol
            company_name: Optional company name for better context

        Returns:
            FundamentalAnalysis object with detailed assessment
        """
        if not self.client:
            logger.warning(f"Perplexity client not available for {symbol}")
            return None

        try:
            logger.info(f"🔍 Analyzing {symbol} fundamentals with Perplexity AI...")

            # Build comprehensive analysis prompt
            prompt = self._build_analysis_prompt(symbol, company_name)

            # Call Perplexity API
            response = await self._call_perplexity_api(prompt)

            if not response:
                logger.warning(f"No response from Perplexity for {symbol}")
                return None

            # Parse and structure the response
            analysis = self._parse_analysis_response(symbol, response)

            logger.info(f"✅ Completed fundamental analysis for {symbol} (Score: {analysis.overall_score:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {symbol} fundamentals: {e}")
            return None

    def _build_analysis_prompt(self, symbol: str, company_name: str = None) -> str:
        """Build detailed analysis prompt for Perplexity"""
        company_ref = f"{company_name} ({symbol})" if company_name else symbol

        prompt = f"""Conduct a comprehensive fundamental analysis of {company_ref} stock. Provide a detailed assessment covering:

1. FINANCIAL HEALTH:
   - Price-to-Earnings (P/E) ratio analysis: Is it overvalued or undervalued compared to industry peers?
   - Price-to-Book (P/B) ratio: Asset valuation assessment
   - Return on Equity (ROE): Profitability and efficiency analysis
   - Debt levels: Total debt, debt-to-equity ratio, and financial leverage assessment
   - Current ratio and liquidity position
   - Free cash flow generation

2. REVENUE AND EARNINGS GROWTH:
   - Historical revenue growth trends (last 3-5 years)
   - Earnings growth patterns and consistency
   - Forward growth projections and analyst estimates
   - Margin trends (gross, operating, net)
   - Growth sustainability and quality of earnings

3. COMPETITIVE POSITION AND MOAT:
   - Economic moat assessment (wide, narrow, or none)
   - Competitive advantages (brand, patents, network effects, cost advantages, switching costs)
   - Market share and positioning vs competitors
   - Barriers to entry in the industry
   - Pricing power

4. MANAGEMENT QUALITY AND GOVERNANCE:
   - Management track record and execution capability
   - Capital allocation decisions
   - Insider ownership and recent insider trading activity
   - Board composition and independence
   - Corporate governance practices
   - Shareholder-friendly policies (dividends, buybacks)

5. INDUSTRY OUTLOOK:
   - Industry growth trends and cyclicality
   - Sector tailwinds and headwinds
   - Regulatory environment and potential changes
   - Technological disruption risks
   - Macroeconomic factors affecting the industry

6. ADDITIONAL CONSIDERATIONS:
   - Institutional ownership and analyst coverage
   - Recent news or developments
   - Key risks to the investment thesis
   - Potential catalysts for stock price appreciation

Provide a structured analysis with:
- Score each category (1-10 scale)
- Overall investment recommendation (Strong Buy, Buy, Hold, Sell, Strong Sell)
- 3-5 key insights
- 3-5 major risks
- 2-3 potential catalysts

Be specific with numbers and comparisons where available."""

        return prompt

    async def _call_perplexity_api(self, prompt: str) -> Optional[str]:
        """
        Call Perplexity API with the analysis prompt
        """
        try:
            # Using sonar-deep-research for comprehensive analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in fundamental stock analysis. Provide detailed, data-driven assessments with specific metrics and comparisons."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,  # Comprehensive response
                temperature=0.3,  # Lower temperature for more factual analysis
            )

            # Extract the response text
            if response and response.choices:
                analysis_text = response.choices[0].message.content
                return analysis_text

            return None

        except Exception as e:
            logger.error(f"Perplexity API call failed: {e}")
            return None

    def _parse_analysis_response(self, symbol: str, response_text: str) -> FundamentalAnalysis:
        """
        Parse Perplexity response into structured FundamentalAnalysis object
        """
        try:
            # Initialize analysis object
            analysis = FundamentalAnalysis(
                symbol=symbol,
                analysis_date=datetime.now(),
                full_analysis_text=response_text,
                competitive_advantages=[],
                key_insights=[],
                risks=[],
                catalysts=[]
            )

            # Parse response text for structured data
            # This is a simplified parser - you can enhance it based on actual response format
            text_lower = response_text.lower()

            # Extract scores from text (looking for patterns like "score: 8/10" or "8 out of 10")
            import re

            # Financial Health Score
            if 'financial health' in text_lower:
                score_match = re.search(r'financial health.*?(\d+(?:\.\d+)?)\s*[/out of]*\s*10', text_lower, re.IGNORECASE)
                if score_match:
                    analysis.financial_health_score = float(score_match.group(1))

            # Growth Score
            if 'growth' in text_lower:
                score_match = re.search(r'growth.*?score.*?(\d+(?:\.\d+)?)\s*[/out of]*\s*10', text_lower, re.IGNORECASE)
                if score_match:
                    analysis.growth_score = float(score_match.group(1))

            # Competitive Score
            if 'competitive' in text_lower or 'moat' in text_lower:
                score_match = re.search(r'(?:competitive|moat).*?(\d+(?:\.\d+)?)\s*[/out of]*\s*10', text_lower, re.IGNORECASE)
                if score_match:
                    analysis.competitive_score = float(score_match.group(1))

            # Management Score
            if 'management' in text_lower:
                score_match = re.search(r'management.*?(\d+(?:\.\d+)?)\s*[/out of]*\s*10', text_lower, re.IGNORECASE)
                if score_match:
                    analysis.management_score = float(score_match.group(1))

            # Industry Score
            if 'industry' in text_lower or 'sector' in text_lower:
                score_match = re.search(r'(?:industry|sector).*?(\d+(?:\.\d+)?)\s*[/out of]*\s*10', text_lower, re.IGNORECASE)
                if score_match:
                    analysis.industry_score = float(score_match.group(1))

            # Extract recommendation
            if 'strong buy' in text_lower:
                analysis.recommendation = 'STRONG_BUY'
            elif 'buy' in text_lower and 'not' not in text_lower.split('buy')[0][-20:]:
                analysis.recommendation = 'BUY'
            elif 'hold' in text_lower:
                analysis.recommendation = 'HOLD'
            elif 'sell' in text_lower:
                analysis.recommendation = 'SELL'

            # Extract key insights (looking for bullet points or numbered lists)
            insights_section = re.search(r'key insights?:?(.*?)(?:risks?:|catalysts?:|$)', text_lower, re.IGNORECASE | re.DOTALL)
            if insights_section:
                insights_text = insights_section.group(1)
                insights = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)', insights_text)
                analysis.key_insights = [insight.strip() for insight in insights[:5]]

            # Extract risks
            risks_section = re.search(r'(?:major )?risks?:?(.*?)(?:catalysts?:|$)', text_lower, re.IGNORECASE | re.DOTALL)
            if risks_section:
                risks_text = risks_section.group(1)
                risks = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)', risks_text)
                analysis.risks = [risk.strip() for risk in risks[:5]]

            # Extract catalysts
            catalysts_section = re.search(r'catalysts?:?(.*?)$', text_lower, re.IGNORECASE | re.DOTALL)
            if catalysts_section:
                catalysts_text = catalysts_section.group(1)
                catalysts = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)', catalysts_text)
                analysis.catalysts = [catalyst.strip() for catalyst in catalysts[:3]]

            # Calculate overall score as weighted average
            scores = []
            weights = []

            if analysis.financial_health_score > 0:
                scores.append(analysis.financial_health_score)
                weights.append(0.25)  # 25% weight

            if analysis.growth_score > 0:
                scores.append(analysis.growth_score)
                weights.append(0.25)  # 25% weight

            if analysis.competitive_score > 0:
                scores.append(analysis.competitive_score)
                weights.append(0.20)  # 20% weight

            if analysis.management_score > 0:
                scores.append(analysis.management_score)
                weights.append(0.15)  # 15% weight

            if analysis.industry_score > 0:
                scores.append(analysis.industry_score)
                weights.append(0.15)  # 15% weight

            if scores:
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]

                # Calculate weighted average
                analysis.overall_score = sum(s * w for s, w in zip(scores, normalized_weights))

            # Extract text assessments for each category
            analysis.pe_ratio_assessment = self._extract_assessment(response_text, ['p/e ratio', 'price-to-earnings'])
            analysis.roe_assessment = self._extract_assessment(response_text, ['return on equity', 'roe'])
            analysis.debt_level_assessment = self._extract_assessment(response_text, ['debt', 'leverage'])
            analysis.revenue_growth_trend = self._extract_assessment(response_text, ['revenue growth'])
            analysis.competitive_moat = self._extract_assessment(response_text, ['moat', 'competitive advantage'])

            return analysis

        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            # Return basic analysis object
            return FundamentalAnalysis(
                symbol=symbol,
                analysis_date=datetime.now(),
                full_analysis_text=response_text,
                overall_score=5.0  # Neutral score as fallback
            )

    def _extract_assessment(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract assessment text for specific keywords"""
        try:
            text_lower = text.lower()
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the sentence containing the keyword
                    start = text_lower.find(keyword)
                    # Find sentence boundaries
                    sentence_start = text_lower.rfind('.', 0, start) + 1
                    sentence_end = text_lower.find('.', start) + 1

                    if sentence_end > sentence_start:
                        sentence = text[sentence_start:sentence_end].strip()
                        return sentence[:200]  # Limit to 200 chars

            return None
        except:
            return None

    async def batch_analyze_stocks(self, stocks: List[Dict[str, str]]) -> Dict[str, FundamentalAnalysis]:
        """
        Analyze multiple stocks in batch (with rate limiting)

        Args:
            stocks: List of dicts with 'symbol' and optional 'name' keys

        Returns:
            Dict mapping symbol to FundamentalAnalysis
        """
        results = {}

        for i, stock in enumerate(stocks, 1):
            symbol = stock['symbol']
            name = stock.get('name')

            logger.info(f"Analyzing {i}/{len(stocks)}: {symbol}")

            analysis = await self.analyze_stock_fundamentals(symbol, name)
            if analysis:
                results[symbol] = analysis

            # Rate limiting: wait between requests
            if i < len(stocks):
                await asyncio.sleep(2)  # 2 second delay between requests

        return results

    def get_fundamental_boost(self, analysis: FundamentalAnalysis) -> float:
        """
        Calculate confidence boost based on fundamental analysis
        Returns a value between -0.15 and +0.15 to adjust ML confidence
        """
        if not analysis:
            return 0.0

        # Convert 0-10 score to -0.15 to +0.15 range
        # Score 5 = neutral (0 boost)
        # Score 10 = max boost (+0.15)
        # Score 0 = max penalty (-0.15)

        normalized_score = (analysis.overall_score - 5) / 5  # -1 to +1
        boost = normalized_score * 0.15  # -0.15 to +0.15

        return max(-0.15, min(0.15, boost))


# Singleton instance
_fundamental_analyzer = None


def get_fundamental_analyzer() -> PerplexityFundamentalAnalyzer:
    """Get singleton fundamental analyzer instance"""
    global _fundamental_analyzer
    if _fundamental_analyzer is None:
        _fundamental_analyzer = PerplexityFundamentalAnalyzer()
    return _fundamental_analyzer
