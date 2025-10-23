# Calculate monthly Perplexity API costs for various models and settings

def calculate_api_cost(
    num_stocks_per_day=50,
    days_per_month=30,
    model="sonar",
    input_tokens_per_stock=3000,
    output_tokens_per_stock=3500
):
    """
    Returns the estimated monthly cost for a given Perplexity model and usage.
    """
    # Recent 2025 prices (adjust if Perplexity changes their pricing)
    PRICING = {
        'sonar': {
            'search_fee_per_1000': 5,
            'input_per_million': 1,
            'output_per_million': 1
        },
        'sonar-pro': {
            'search_fee_per_1000': 5,
            'input_per_million': 3,
            'output_per_million': 15
        },
        'deep-research': {
            'search_fee_per_1000': 5,
            'input_per_million': 8,
            'output_per_million': 30
        }
    }
    m = model.lower()
    if m not in PRICING:
        raise ValueError(f"Unknown model '{model}', choose from {list(PRICING.keys())}")

    pricing = PRICING[m]
    total_queries = num_stocks_per_day * days_per_month
    total_input_tokens = input_tokens_per_stock * total_queries
    total_output_tokens = output_tokens_per_stock * total_queries

    search_fee = (total_queries / 1000) * pricing['search_fee_per_1000']
    input_fee = (total_input_tokens / 1_000_000) * pricing['input_per_million']
    output_fee = (total_output_tokens / 1_000_000) * pricing['output_per_million']
    total_cost = search_fee + input_fee + output_fee

    print(f"\n===== Perplexity API Cost Estimate =====")
    print(f"Model:         {model}")
    print(f"Stocks/Day:    {num_stocks_per_day}")
    print(f"Days/Month:    {days_per_month}")
    print(f"Tokens/Query:  input={input_tokens_per_stock}, output={output_tokens_per_stock}")
    print(f"----------------------------------------")
    print(f"Monthly Queries:   {total_queries}")
    print(f"Total Input Tokens:  {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"----------------------------------------")
    print(f"Search Fee:    ${search_fee:.2f}")
    print(f"Input Tokens:  ${input_fee:.2f}")
    print(f"Output Tokens: ${output_fee:.2f}")
    print(f"----------------------------------------")
    print(f"Estimated Total Monthly Cost: ${total_cost:.2f}")
    print("========================================\n")
    return total_cost

# Example usage:
if __name__ == "__main__":
    # Default: 10 stocks/day, 30 days, default tokens per stock
    calculate_api_cost(model="sonar")

    # Sonar-Pro example
    calculate_api_cost(model="sonar-pro")

    # Deep Research example, adjust tokens for bigger output
    calculate_api_cost(model="deep-research", input_tokens_per_stock=3500, output_tokens_per_stock=6000)
