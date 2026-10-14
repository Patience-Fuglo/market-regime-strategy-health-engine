# market-regime-strategy-health-engine

Answers: "what market regime is present, is a trading strategy
behaving as expected, and if not, is the deterioration coming from
changing market conditions, weakened alpha, factor exposure drift,
increased transaction costs, or a data problem?" A transparent,
threshold-based market regime classifier (volatility, trend,
liquidity, correlation); real strategy performance monitoring
(drawdown, rolling Sharpe, turnover, cost); real signal health
monitoring (information coefficient, multi-horizon decay, stability,
exposure drift); regime-conditioned performance diagnostics (an
association, never a causal claim); a diagnostic framework that
separates five real deterioration sources; and configurable,
severity-tagged alerts with real evidence behind every one.

This engine never re-estimates a factor exposure, factor covariance,
optimization result, portfolio risk figure, or transaction cost -- it
consumes `factor-model-engine`'s, `portfolio-construction-engine`'s,
`portfolio-risk-engine`'s, and `transaction-cost-engine`'s own real
outputs directly. In particular, its signal-health diagnostic reuses
`factor-model-engine`'s own real `monitoring.compute_factor_stability`
(the same point-in-time, cross-sectional IC/ICIR/drift check that
module already performs) rather than re-deriving a parallel one, and
adds only what that module does not already provide: a real
multi-horizon decay curve and portfolio-level exposure drift.

**All 10 stages are complete.** Point-in-time data ingestion/health
(reusing `factor-model-engine`'s own real diagnostics); real market
regime features and a transparent rule-based classifier with explicit
`UNKNOWN`/`AMBIGUOUS` states; real strategy performance monitoring;
real signal health monitoring; regime-conditioned performance
diagnostics; a five-category root-cause diagnostic framework;
configurable severity-tiered alerts; one consolidated end-to-end
pipeline; a reproducible JSON/Markdown report; and a comprehensive
hardening pass confirming exact reproducibility and a real
point-in-time no-look-ahead property across the whole pipeline. See
[`docs/STATUS.md`](docs/STATUS.md) for the full stage-by-stage build
history, final architecture/formula summary, test evidence, and an
honest list of limitations.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For a byte-reproducible install (the exact dependency versions this
repo was actually tested against -- see `constraints.txt`):

```bash
pip install -e ".[dev]" -c constraints.txt
```

## Run

```bash
python -m pytest tests/ -v
python scripts/run_regime_health_report.py
```

## Repository layout

```
src/market_regime_health/
  data/schemas.py                              # RegimeFeatureRow, RegimeClassification,
                                                  # StrategyPerformanceSnapshot, SignalHealthResult,
                                                  # RegimeConditionedResult, DiagnosticFinding, Alert
  regime/
    features.py                                   # compute_regime_features() -- vol/trend/liquidity/correlation
    classification.py                              # classify_regime() -- transparent rule-based labels
  strategy_health/
    performance.py                                   # build_strategy_performance_snapshots() -- drawdown, rolling Sharpe
    signal_health.py                                  # compute_signal_health() -- IC/ICIR reuse, decay curve, exposure drift
  regime_conditioned/
    conditioned_performance.py                          # compute_regime_conditioned_performance()
  diagnostics/
    diagnostic_engine.py                                  # run_diagnostics() -- 5 real, separate deterioration categories
  alerts/
    alerts.py                                               # generate_alerts() -- severity-tiered, evidenced
  pipeline/pipeline.py                                       # run_full_regime_health_analysis() -- Stages 1-8, end to end
  reporting/report.py                                         # report_to_dict()/render_markdown_report()

tests/
  unit/           # one file per module, including every rule-path in the classifier
  integration/     # full synthetic-dataset pipeline, reproducibility + no-look-ahead
                    # hardening, real CLI subprocess test

scripts/
  run_regime_health_report.py   # end-to-end: synthetic data -> rebalance -> regime/health analysis -> report
```
