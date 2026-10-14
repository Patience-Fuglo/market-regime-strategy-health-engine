#!/usr/bin/env python3
"""End-to-end CLI: generates a real synthetic dataset, runs a real
monthly rebalancing schedule with portfolio-construction-engine, then
runs this engine's full market-regime-and-strategy-health analysis and
writes one reproducible JSON + Markdown report.

Usage:
    python scripts/run_regime_health_report.py [--n-securities 20] [--n-periods-months 48] [--signal-factor momentum] [--output-dir outputs/reports]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from factor_model.config import EngineConfig as FactorModelConfig, UniverseConfig
from factor_model.data import SyntheticDataConfig, fundamental_observations_from_frame, generate_synthetic_dataset, market_observations_from_frame
from portfolio_construction.config import EngineConfig as PortfolioConstructionConfig
from portfolio_construction.constraints import GrossLeverageConstraint, MaxPositionWeightConstraint
from portfolio_construction.rebalancing import run_rebalancing_schedule

from market_regime_health.config import EngineConfig
from market_regime_health.pipeline import run_full_regime_health_analysis
from market_regime_health.reporting import render_markdown_report, report_to_dict


def _zero_expected_returns(as_of, security_ids):
    return {sid: 0.0 for sid in security_ids}


def _standard_constraints(as_of, exposure_matrix):
    return [MaxPositionWeightConstraint(max_weight=0.2, long_only=True), GrossLeverageConstraint(max_gross_leverage=1.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-securities", type=int, default=20)
    parser.add_argument("--n-periods-months", type=int, default=48)
    parser.add_argument("--signal-factor", default="momentum")
    parser.add_argument("--portfolio-id", default="portfolio")
    parser.add_argument("--output-dir", default="outputs/reports")
    args = parser.parse_args()

    data_cfg = SyntheticDataConfig(n_securities=args.n_securities, n_sectors=5, n_periods_months=args.n_periods_months)
    dataset = generate_synthetic_dataset(data_cfg)
    market_obs = market_observations_from_frame(dataset.market)
    fundamental_obs = fundamental_observations_from_frame(dataset.fundamentals)
    dates = sorted(dataset.market["as_of"].unique())
    period_dates = [pd.Timestamp(d) for d in dates]

    fm_cfg = FactorModelConfig(universe=UniverseConfig(min_market_cap=0.0, min_adv_usd=0.0))
    pc_cfg = PortfolioConstructionConfig()
    schedule = run_rebalancing_schedule(
        market_obs, fundamental_obs, period_dates[12:], lookback_periods=12, objective="min_variance",
        expected_returns_source=_zero_expected_returns, constraints_factory=_standard_constraints,
        fm_engine_cfg=fm_cfg, pc_engine_cfg=pc_cfg,
    )

    engine_cfg = EngineConfig()
    as_of = schedule.results[-1].as_of
    result = run_full_regime_health_analysis(
        market_obs, fundamental_obs, schedule.results, period_dates, as_of, args.signal_factor, fm_cfg, engine_cfg,
        portfolio_id=args.portfolio_id,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"regime_health_report_{args.portfolio_id}.json"
    md_path = out_dir / f"regime_health_report_{args.portfolio_id}.md"

    with open(json_path, "w") as f:
        json.dump(report_to_dict(result), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(md_path, "w") as f:
        f.write(render_markdown_report(result))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"regime: {result.regime_classification.label.value}")
    print(f"net return: {result.strategy_performance.net_return:.4%}  drawdown: {result.strategy_performance.drawdown:.4%}")
    print(f"{len(result.findings)} real diagnostic findings, {len(result.alerts)} real alerts")
    for alert in result.alerts:
        print(f"  - [{alert.severity.upper()}] {alert.message}")


if __name__ == "__main__":
    main()
