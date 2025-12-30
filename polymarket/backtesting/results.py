"""
Backtest results and simulated trade dataclasses.

These classes track the results of backtesting runs and provide
comprehensive reporting with bias warnings.

Includes hedge metrics for strategies with hedge simulation.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .strategies.bond_backtest import SimulatedHedgeTrade


@dataclass
class SimulatedTrade:
    """A simulated trade from backtesting"""
    market_question: str
    token_id: str
    token_outcome: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    shares: float
    cost: float
    proceeds: Optional[float]
    pnl: Optional[float]
    pnl_percent: Optional[float]
    resolved_to: Optional[float]  # 1.0 or 0.0
    held_to_resolution: bool
    reason: str  # Why trade was made
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl is not None and self.pnl > 0
    
    @property
    def is_complete(self) -> bool:
        """Check if trade has been exited"""
        return self.exit_time is not None


@dataclass
class BacktestResults:
    """Comprehensive backtest results with bias warnings"""
    
    # Configuration
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # Results
    final_capital: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Financial metrics
    total_pnl: float = 0.0
    total_fees: float = 0.0
    gross_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    
    # Trade list
    trades: List[SimulatedTrade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: List[tuple] = field(default_factory=list)  # [(timestamp, equity), ...]
    
    # Bias warnings
    survivorship_bias_warning: bool = True  # Always true for closed markets
    look_ahead_bias_warning: bool = False   # Resolution data from API, not price inference
    execution_assumption_warning: bool = True  # Always true for simulated execution
    
    # Data quality info
    uses_price_history: bool = True   # /prices-history endpoint
    uses_orderbook_depth: bool = False  # No historical orderbook depth available
    
    # Spread/liquidity estimation method
    spread_estimation_method: str = "price_bounce"  # Method used to estimate spread
    liquidity_estimation_method: str = "multi_window_volatility"  # Method used to estimate liquidity
    
    # Additional metadata
    markets_analyzed: int = 0
    markets_traded: int = 0
    
    # Hedge metrics
    hedge_trades: List[Any] = field(default_factory=list)  # List of SimulatedHedgeTrade
    hedges_triggered: int = 0
    hedge_pnl: float = 0.0  # P&L from hedge trades
    loss_avoided_by_hedging: float = 0.0  # Estimated loss avoided
    
    @property
    def win_rate(self) -> float:
        """Win rate as percentage"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def loss_rate(self) -> float:
        """Loss rate as percentage"""
        if self.total_trades == 0:
            return 0.0
        return self.losing_trades / self.total_trades
    
    @property
    def return_pct(self) -> float:
        """Total return as percentage"""
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital
    
    @property
    def avg_trade_pnl(self) -> float:
        """Average P&L per trade"""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def avg_winner(self) -> float:
        """Average winning trade P&L"""
        winners = [t.pnl for t in self.trades if t.pnl and t.pnl > 0]
        return statistics.mean(winners) if winners else 0.0
    
    @property
    def avg_loser(self) -> float:
        """Average losing trade P&L"""
        losers = [t.pnl for t in self.trades if t.pnl and t.pnl < 0]
        return statistics.mean(losers) if losers else 0.0
    
    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss"""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def sharpe_ratio(self) -> Optional[float]:
        """Sharpe ratio (assuming 0% risk-free rate)"""
        if len(self.trades) < 10:
            return None
        
        returns = [t.pnl_percent for t in self.trades if t.pnl_percent is not None]
        if len(returns) < 2:
            return None
        
        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return None
            
            # Annualize (assuming ~250 trading days)
            return (mean_return * 250) / (std_return * (250 ** 0.5))
        except statistics.StatisticsError:
            return None
    
    def add_trade(self, trade: SimulatedTrade):
        """Add a trade and update statistics"""
        self.trades.append(trade)
        self.total_trades += 1
        
        if trade.pnl:
            self.total_pnl += trade.pnl
            if trade.pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
    
    def finalize(self):
        """Calculate final metrics after all trades added"""
        self.final_capital = self.initial_capital + self.total_pnl - self.total_fees
        
        # Calculate max drawdown from equity curve
        if self.equity_curve:
            peak = self.initial_capital
            max_dd = 0.0
            
            for _, equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            
            self.max_drawdown = max_dd
            self.peak_equity = peak
            
            if peak > 0:
                self.max_drawdown_pct = max_dd / peak
    
    def print_report(self):
        """Print comprehensive results report"""
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {self.strategy_name}")
        print("="*60)
        
        print(f"\nPeriod: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Markets Analyzed: {self.markets_analyzed}")
        print(f"Markets Traded: {self.markets_traded}")
        
        print("\n--- PERFORMANCE ---")
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${self.final_capital:,.2f}")
        print(f"Total P&L:        ${self.total_pnl:,.2f} ({self.return_pct:.1%})")
        print(f"Fees Paid:        ${self.total_fees:,.2f}")
        
        print("\n--- TRADES ---")
        print(f"Total Trades:     {self.total_trades}")
        print(f"Winning Trades:   {self.winning_trades} ({self.win_rate:.1%})")
        print(f"Losing Trades:    {self.losing_trades} ({self.loss_rate:.1%})")
        print(f"Avg Trade P&L:    ${self.avg_trade_pnl:.2f}")
        print(f"Avg Winner:       ${self.avg_winner:.2f}")
        print(f"Avg Loser:        ${self.avg_loser:.2f}")
        print(f"Profit Factor:    {self.profit_factor:.2f}")
        
        print("\n--- RISK ---")
        print(f"Max Drawdown:     ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.1%})")
        print(f"Peak Equity:      ${self.peak_equity:,.2f}")
        
        sharpe = self.sharpe_ratio
        if sharpe is not None:
            print(f"Sharpe Ratio:     {sharpe:.2f}")
        else:
            print("Sharpe Ratio:     N/A (insufficient data)")
        
        # Hedge metrics if any hedges were triggered
        if self.hedges_triggered > 0:
            print("\n--- HEDGE ACTIVITY ---")
            print(f"Hedges Triggered: {self.hedges_triggered}")
            print(f"Hedge Trades:     {len(self.hedge_trades)}")
            print(f"Hedge P&L:        ${self.hedge_pnl:,.2f}")
            print(f"Loss Avoided:     ${self.loss_avoided_by_hedging:,.2f}")
            
            # Breakdown by hedge type
            from collections import Counter
            if self.hedge_trades:
                type_counts = Counter(str(ht.action.value) if hasattr(ht, 'action') else 'unknown' for ht in self.hedge_trades)
                print(f"Hedge Types:      {dict(type_counts)}")
        
        # Detailed trade listing
        if self.trades:
            print("\n--- TRADE DETAILS ---")
            print(f"Total Trades: {len(self.trades)}")
            # Print first 10 trades summary, full details available via print_trade_details()
            print("\nFirst 10 trades:")
            for i, trade in enumerate(self.trades[:10], 1):
                pnl_str = f"${trade.pnl:,.2f}" if trade.pnl is not None else "N/A"
                pnl_pct_str = f"({trade.pnl_percent:.1%})" if trade.pnl_percent is not None else ""
                print(f"  [{i}] {trade.market_question[:50]}...")
                print(f"      Entry: ${trade.entry_price:.4f} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if trade.exit_time:
                    print(f"      Exit:  ${trade.exit_price:.4f} @ {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"      P&L:   {pnl_str} {pnl_pct_str}")
                print(f"      Reason: {trade.reason[:60]}")
            if len(self.trades) > 10:
                print(f"\n  ... and {len(self.trades) - 10} more trades (call results.print_trade_details() for full list)")
        
        # BIAS WARNINGS
        print("\n" + "!"*60)
        print("IMPORTANT WARNINGS")
        print("!"*60)
        
        if self.survivorship_bias_warning:
            print("\n⚠️  SURVIVORSHIP BIAS:")
            print("   Only resolved markets were analyzed.")
            print("   Cancelled/disputed markets not included.")
        
        if self.look_ahead_bias_warning:
            print("\n⚠️  LOOK-AHEAD BIAS:")
            print("   Some future data may have been used in analysis.")
        
        # Data quality note
        if self.uses_price_history:
            print("\n✅ HISTORICAL DATA:")
            print("   Using /prices-history timeseries endpoint.")
            print("   Resolution determined from API outcomePrices (no look-ahead bias).")
            if not self.uses_orderbook_depth:
                print("\n⚠️  NO HISTORICAL ORDERBOOK:")
                print(f"   Spread estimation method: {self.spread_estimation_method}")
                print(f"   Liquidity estimation method: {self.liquidity_estimation_method}")
                print("   These are approximations based on price patterns.")
        
        if self.execution_assumption_warning:
            print("\n⚠️  EXECUTION ASSUMPTIONS:")
            print("   Slippage modeled based on estimated liquidity.")
            print("   Spread estimated from price bounce patterns.")
            print("   Real slippage may vary with market conditions.")
            print("   Large orders may have more market impact.")
            print("   Partial fills not accurately modeled without orderbook.")
        
        print("\n" + "="*60)
        print("⚠️  ACTUAL RESULTS MAY VARY SIGNIFICANTLY")
        print("="*60 + "\n")
    
    def print_trade_details(self, limit: Optional[int] = None):
        """Print comprehensive trade-by-trade breakdown"""
        if not self.trades:
            print("No trades to display.")
            return
        
        print("\n" + "="*80)
        print("DETAILED TRADE LISTING")
        print("="*80)
        
        trades_to_show = self.trades[:limit] if limit else self.trades
        
        for i, trade in enumerate(trades_to_show, 1):
            print(f"\n--- Trade #{i} ---")
            print(f"Market:     {trade.market_question[:70]}")
            print(f"Token:      {trade.token_id[:16]}... ({trade.token_outcome})")
            print(f"Entry:      ${trade.entry_price:.4f} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Shares:     {trade.shares:.4f}")
            print(f"Cost:       ${trade.cost:,.2f}")
            
            if trade.exit_time:
                print(f"Exit:       ${trade.exit_price:.4f} @ {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                if trade.proceeds:
                    print(f"Proceeds:   ${trade.proceeds:,.2f}")
                
                if trade.pnl is not None:
                    pnl_sign = "+" if trade.pnl >= 0 else ""
                    print(f"P&L:        {pnl_sign}${trade.pnl:,.2f}", end="")
                    if trade.pnl_percent is not None:
                        print(f" ({pnl_sign}{trade.pnl_percent:.2%})")
                    else:
                        print()
                
                if trade.resolved_to is not None:
                    print(f"Resolved:   ${trade.resolved_to:.2f}")
                    print(f"Held to resolution: {'Yes' if trade.held_to_resolution else 'No'}")
                
                # Calculate duration
                duration = trade.exit_time - trade.entry_time
                if duration.total_seconds() < 3600:
                    duration_str = f"{duration.total_seconds() / 60:.1f} minutes"
                elif duration.total_seconds() < 86400:
                    duration_str = f"{duration.total_seconds() / 3600:.1f} hours"
                else:
                    duration_str = f"{duration.days} days"
                print(f"Duration:   {duration_str}")
            else:
                print("Exit:       Not yet exited")
            
            print(f"Reason:     {trade.reason}")
            print(f"Winner:     {'Yes' if trade.is_winner else 'No'}")
        
        if limit and len(self.trades) > limit:
            print(f"\n... and {len(self.trades) - limit} more trades")
        
        print("\n" + "="*80)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export"""
        result = {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_trade_pnl": self.avg_trade_pnl,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "total_fees": self.total_fees,
            "markets_analyzed": self.markets_analyzed,
            "markets_traded": self.markets_traded,
            "warnings": {
                "survivorship_bias": self.survivorship_bias_warning,
                "look_ahead_bias": self.look_ahead_bias_warning,
                "execution_assumption": self.execution_assumption_warning,
            },
            "estimation_methods": {
                "spread": self.spread_estimation_method,
                "liquidity": self.liquidity_estimation_method,
                "uses_orderbook_depth": self.uses_orderbook_depth,
            }
        }
        
        # Add hedge metrics if hedging was used
        if self.hedges_triggered > 0 or self.hedge_trades:
            from collections import Counter
            type_counts = Counter(
                str(ht.action.value) if hasattr(ht, 'action') else 'unknown' 
                for ht in self.hedge_trades
            )
            result["hedge_metrics"] = {
                "hedges_triggered": self.hedges_triggered,
                "hedge_trades": len(self.hedge_trades),
                "hedge_pnl": self.hedge_pnl,
                "loss_avoided": self.loss_avoided_by_hedging,
                "hedge_type_breakdown": dict(type_counts),
            }
        
        return result

