#!/usr/bin/env python3
"""
Test fetching resolved markets with proper resolution detection.

Tries multiple approaches:
1. Closed markets with resolved=true from Gamma API
2. Events endpoint for recently resolved markets
3. Active markets with prices at terminal values AND end_date passed
"""

import asyncio
import aiohttp
import ssl
import certifi
import json
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"


def get_ssl_context():
    """Create SSL context with proper certificates."""
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return ssl_ctx


def is_resolved_from_prices(prices, end_date_str):
    """Check if a market is truly resolved based on prices and end date."""
    if not prices:
        return False, None

    parsed_prices = []
    for p in prices:
        try:
            parsed_prices.append(float(p))
        except:
            parsed_prices.append(None)

    # Check if end date has passed
    end_passed = False
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            end_passed = end_date < datetime.now(timezone.utc)
        except:
            pass

    # Terminal prices: exactly 1.0 (or 0.9999+) and 0.0 (or 0.0001-)
    has_winner = any(p is not None and p >= 0.9999 for p in parsed_prices)
    has_loser = any(p is not None and p <= 0.0001 for p in parsed_prices)

    # Find winner index
    winner_idx = None
    if has_winner and has_loser:
        for i, p in enumerate(parsed_prices):
            if p is not None and p >= 0.9999:
                winner_idx = i
                break

    return has_winner and has_loser and end_passed, winner_idx


def parse_json_field(value):
    """Parse a field that might be JSON string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            return []
    return []


async def approach_1_closed_resolved(session):
    """Approach 1: Get closed markets with resolved=true flag."""
    logger.info("APPROACH 1: Fetching closed+resolved markets...")

    resolved_markets = []
    offset = 0

    while offset < 1000:
        url = f"{GAMMA_API}/markets"
        params = {
            "limit": 100,
            "offset": offset,
            "closed": "true",
            "resolved": "true",
            "order": "endDate",
            "ascending": "false",  # Most recent first
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    logger.error(f"Failed: {resp.status}")
                    break

                data = await resp.json()
                if not data:
                    break

                for market in data:
                    # Parse end date
                    end_str = market.get('endDate')
                    end_date = None
                    if end_str:
                        try:
                            end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                        except:
                            pass

                    # Skip markets older than 90 days
                    if end_date:
                        age_days = (datetime.now(timezone.utc) - end_date).days
                        if age_days > 90:
                            continue

                    token_ids = parse_json_field(market.get('clobTokenIds', []))
                    outcomes = parse_json_field(market.get('outcomes', []))
                    prices = parse_json_field(market.get('outcomePrices', []))

                    # Get winning outcome
                    winning_outcome = market.get('winningOutcome')

                    if not winning_outcome and prices:
                        _, winner_idx = is_resolved_from_prices(prices, end_str)
                        if winner_idx is not None and winner_idx < len(outcomes):
                            winning_outcome = outcomes[winner_idx]

                    if token_ids and winning_outcome:
                        resolved_markets.append({
                            'condition_id': market.get('conditionId'),
                            'question': market.get('question', '')[:80],
                            'winning_outcome': winning_outcome,
                            'token_ids': token_ids,
                            'outcomes': outcomes,
                            'prices': prices,
                            'end_date': end_str,
                            'source': 'closed+resolved',
                        })

                if len(data) < 100:
                    break
                offset += 100

        except Exception as e:
            logger.error(f"Error in approach 1: {e}")
            break

    logger.info(f"Approach 1 found: {len(resolved_markets)} markets")
    return resolved_markets


async def approach_2_events(session):
    """Approach 2: Check events/resolutions endpoint."""
    logger.info("APPROACH 2: Checking events endpoint...")

    # Polymarket might have an events or resolutions endpoint
    resolved_markets = []

    # Try the events endpoint
    try:
        url = f"{GAMMA_API}/events"
        params = {
            "limit": 100,
            "order": "endDate",
            "ascending": "false",
        }

        async with session.get(url, params=params, timeout=30) as resp:
            if resp.status == 200:
                data = await resp.json()
                logger.info(f"Events endpoint returned {len(data) if isinstance(data, list) else 'non-list'}")

                if isinstance(data, list):
                    for event in data[:5]:
                        logger.info(f"  Event: {event.get('title', 'No title')[:50]}...")
    except Exception as e:
        logger.debug(f"Events endpoint failed: {e}")

    return resolved_markets


async def approach_3_recent_active(session):
    """Approach 3: Check active markets for those with past end_date and terminal prices."""
    logger.info("APPROACH 3: Checking active markets with past end_date...")

    resolved_markets = []
    offset = 0

    while offset < 1000:
        url = f"{GAMMA_API}/markets"
        params = {
            "limit": 100,
            "offset": offset,
            "active": "true",
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    break

                data = await resp.json()
                if not data:
                    break

                for market in data:
                    end_str = market.get('endDate')
                    prices = parse_json_field(market.get('outcomePrices', []))

                    is_resolved, winner_idx = is_resolved_from_prices(prices, end_str)

                    if is_resolved:
                        token_ids = parse_json_field(market.get('clobTokenIds', []))
                        outcomes = parse_json_field(market.get('outcomes', []))

                        winning_outcome = None
                        if winner_idx is not None and winner_idx < len(outcomes):
                            winning_outcome = outcomes[winner_idx]

                        if token_ids and winning_outcome:
                            resolved_markets.append({
                                'condition_id': market.get('conditionId'),
                                'question': market.get('question', '')[:80],
                                'winning_outcome': winning_outcome,
                                'token_ids': token_ids,
                                'outcomes': outcomes,
                                'prices': prices,
                                'end_date': end_str,
                                'source': 'active_but_resolved',
                            })

                if len(data) < 100:
                    break
                offset += 100

        except Exception as e:
            logger.error(f"Error in approach 3: {e}")
            break

    logger.info(f"Approach 3 found: {len(resolved_markets)} markets")
    return resolved_markets


async def approach_4_all_markets(session):
    """Approach 4: Fetch ALL markets and filter for recently resolved."""
    logger.info("APPROACH 4: Fetching all markets (no filters)...")

    all_markets = []
    resolved_markets = []
    offset = 0

    while offset < 2000:
        url = f"{GAMMA_API}/markets"
        params = {
            "limit": 100,
            "offset": offset,
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    break

                data = await resp.json()
                if not data:
                    break

                all_markets.extend(data)

                if len(data) < 100:
                    break
                offset += 100

        except Exception as e:
            logger.error(f"Error: {e}")
            break

    logger.info(f"Fetched {len(all_markets)} total markets")

    # Filter for resolved markets with recent end dates
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=60)

    for market in all_markets:
        end_str = market.get('endDate')
        if not end_str:
            continue

        try:
            end_date = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        except:
            continue

        # Only look at markets that ended in last 60 days
        if end_date < cutoff or end_date > now:
            continue

        # Check if market is resolved
        is_closed = market.get('closed', False)
        is_resolved_flag = market.get('resolved', False)

        prices = parse_json_field(market.get('outcomePrices', []))
        is_resolved_prices, winner_idx = is_resolved_from_prices(prices, end_str)

        if is_resolved_flag or is_resolved_prices:
            token_ids = parse_json_field(market.get('clobTokenIds', []))
            outcomes = parse_json_field(market.get('outcomes', []))

            winning_outcome = market.get('winningOutcome')
            if not winning_outcome and winner_idx is not None and winner_idx < len(outcomes):
                winning_outcome = outcomes[winner_idx]

            if token_ids and winning_outcome:
                resolved_markets.append({
                    'condition_id': market.get('conditionId'),
                    'question': market.get('question', '')[:80],
                    'winning_outcome': winning_outcome,
                    'token_ids': token_ids,
                    'outcomes': outcomes,
                    'prices': prices,
                    'end_date': end_str,
                    'source': 'all_markets_filter',
                    'closed': is_closed,
                    'resolved_flag': is_resolved_flag,
                })

    logger.info(f"Approach 4 found: {len(resolved_markets)} markets")
    return resolved_markets


async def check_price_history(session, resolved_markets, limit=20):
    """Check which resolved markets have price history."""
    logger.info(f"\nChecking price history for up to {limit} markets...")

    markets_with_history = []

    for m in resolved_markets[:limit]:
        if not m['token_ids']:
            continue

        token_id = m['token_ids'][0]

        url = f"{CLOB_HOST}/prices-history"
        params = {"market": token_id, "interval": "1h"}

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    history = data.get('history', [])

                    if len(history) >= 20:
                        m['price_history_points'] = len(history)
                        markets_with_history.append(m)
                        logger.info(f"+ {m['question'][:50]}... ({len(history)} pts)")
                    else:
                        logger.info(f"- {m['question'][:50]}... (only {len(history)} pts)")
        except Exception as e:
            logger.debug(f"Error: {e}")

    return markets_with_history


async def main():
    ssl_ctx = get_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(connector=connector) as session:
        all_resolved = []

        # Try all approaches
        results1 = await approach_1_closed_resolved(session)
        all_resolved.extend(results1)

        results2 = await approach_2_events(session)
        all_resolved.extend(results2)

        results3 = await approach_3_recent_active(session)
        all_resolved.extend(results3)

        results4 = await approach_4_all_markets(session)
        all_resolved.extend(results4)

        # Deduplicate by condition_id
        seen = set()
        unique_resolved = []
        for m in all_resolved:
            if m['condition_id'] not in seen:
                seen.add(m['condition_id'])
                unique_resolved.append(m)

        print("\n" + "="*80)
        print(f"TOTAL UNIQUE RESOLVED MARKETS: {len(unique_resolved)}")
        print("="*80)

        # Print first 10
        for i, m in enumerate(unique_resolved[:10]):
            print(f"\n{i+1}. {m['question']}")
            print(f"   Winner: {m['winning_outcome']}")
            print(f"   End: {m['end_date']}")
            print(f"   Source: {m['source']}")

        # Check price history
        print("\n" + "="*80)
        print("CHECKING PRICE HISTORY")
        print("="*80)

        markets_with_history = await check_price_history(session, unique_resolved, limit=30)

        print("\n" + "="*80)
        print(f"MARKETS WITH PRICE HISTORY: {len(markets_with_history)}")
        print("="*80)

        for m in markets_with_history[:5]:
            print(f"  - {m['question']}")
            print(f"    Winner: {m['winning_outcome']}, {m['price_history_points']} price points")

        return markets_with_history


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nFound {len(result)} resolved markets with price history")
