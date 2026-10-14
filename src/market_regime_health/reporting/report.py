"""Stage 10 -- renders a real ``RegimeHealthAnalysisResult`` into a
reproducible report: a JSON-serializable dict, and a professional
Markdown document. Both derived from the SAME object -- never
recomputed separately, so they can't silently disagree. Mirrors the
other four engines' own reporting convention.
"""

from __future__ import annotations

from ..pipeline import RegimeHealthAnalysisResult


def report_to_dict(result: RegimeHealthAnalysisResult) -> dict:
    return {
        "as_of": str(result.as_of.date()),
        "portfolio_id": result.portfolio_id,
        "signal_factor_name": result.signal_factor_name,
        "regime": {
            "label": result.regime_classification.label.value,
            "triggered_rules": list(result.regime_classification.triggered_rules),
            "evidence": result.regime_classification.evidence,
        },
        "strategy_performance": {
            "gross_return": result.strategy_performance.gross_return,
            "net_return": result.strategy_performance.net_return,
            "cumulative_net_return": result.strategy_performance.cumulative_net_return,
            "drawdown": result.strategy_performance.drawdown,
            "rolling_sharpe": result.strategy_performance.rolling_sharpe,
            "turnover": result.strategy_performance.turnover,
            "transaction_cost": result.strategy_performance.transaction_cost,
        },
        "signal_health": {
            "historical_ic_mean": result.signal_health.historical_ic_mean,
            "recent_ic_mean": result.signal_health.recent_ic_mean,
            "icir": result.signal_health.icir,
            "decay_curve": result.signal_health.decay_curve,
            "exposure_drift": result.signal_health.exposure_drift,
            "n_periods": result.signal_health.n_periods,
        },
        "regime_conditioned": [
            {
                "regime": r.regime.value,
                "n_periods": r.n_periods,
                "mean_net_return": r.mean_net_return,
                "volatility_of_net_return": r.volatility_of_net_return,
                "sharpe": r.sharpe,
                "is_association_not_causation": r.is_association_not_causation,
            }
            for r in result.regime_conditioned
        ],
        "data_health_warnings": list(result.data_health.warnings),
        "findings": [
            {"category": f.category.value, "description": f.description, "evidence": f.evidence, "is_causal_claim": f.is_causal_claim}
            for f in result.findings
        ],
        "alerts": [
            {"category": a.category.value, "severity": a.severity, "message": a.message, "evidence": a.evidence}
            for a in result.alerts
        ],
    }


def render_markdown_report(result: RegimeHealthAnalysisResult) -> str:
    payload = report_to_dict(result)
    lines: list[str] = []
    lines.append("# Market Regime & Strategy Health -- Research Report")
    lines.append("")
    lines.append(f"**As of:** {payload['as_of']}  **Portfolio:** `{payload['portfolio_id']}`  **Signal:** `{payload['signal_factor_name']}`")
    lines.append("")
    lines.append("All data in this report is synthetic and non-proprietary.")
    lines.append("")

    regime = payload["regime"]
    lines.append("## Market Regime")
    lines.append(f"- Label: **{regime['label']}**")
    lines.append(f"- Triggered rules: {', '.join(regime['triggered_rules']) or 'none'}")
    for name, value in regime["evidence"].items():
        lines.append(f"  - {name}: {value:.4f}")
    lines.append("")

    perf = payload["strategy_performance"]
    lines.append("## Strategy Performance")
    lines.append(f"- Gross return: {perf['gross_return']:.4%}  Net return: {perf['net_return']:.4%}")
    lines.append(f"- Cumulative net return: {perf['cumulative_net_return']:.4%}  Drawdown: {perf['drawdown']:.4%}")
    rolling_sharpe = f"{perf['rolling_sharpe']:.2f}" if perf["rolling_sharpe"] is not None else "n/a"
    lines.append(f"- Rolling Sharpe: {rolling_sharpe}  Turnover: {perf['turnover']:.4f}  Transaction cost: {perf['transaction_cost']:.4%}")
    lines.append("")

    sig = payload["signal_health"]
    lines.append("## Signal Health")
    lines.append(f"- Historical IC mean: {sig['historical_ic_mean']:.4f}  Recent IC mean: {sig['recent_ic_mean']:.4f}  N periods: {sig['n_periods']}")
    lines.append(f"- ICIR: {sig['icir']:.4f}")
    if sig["decay_curve"]:
        lines.append("- Decay curve: " + ", ".join(f"h={h}: {v:.4f}" for h, v in sorted(sig["decay_curve"].items())))
    if sig["exposure_drift"]:
        lines.append("- Exposure drift: " + ", ".join(f"{name}: {v:.4f}" for name, v in sig["exposure_drift"].items()))
    lines.append("")

    if payload["regime_conditioned"]:
        lines.append("## Regime-Conditioned Performance (association, not causation)")
        lines.append("| Regime | N periods | Mean net return | Vol | Sharpe |")
        lines.append("|---|---|---|---|---|")
        for r in payload["regime_conditioned"]:
            sharpe = f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "n/a"
            lines.append(f"| {r['regime']} | {r['n_periods']} | {r['mean_net_return']:.4%} | {r['volatility_of_net_return']:.4%} | {sharpe} |")
        lines.append("")

    lines.append("## Diagnostic Findings")
    if payload["findings"]:
        for f in payload["findings"]:
            lines.append(f"- **[{f['category']}]** {f['description']}")
    else:
        lines.append("No real diagnostic findings.")
    lines.append("")

    lines.append("## Alerts")
    if payload["alerts"]:
        for a in payload["alerts"]:
            lines.append(f"- **[{a['severity'].upper()}][{a['category']}]** {a['message']}")
    else:
        lines.append("No real alerts.")
    lines.append("")

    if payload["data_health_warnings"]:
        lines.append("## Data Health Warnings")
        for w in payload["data_health_warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)
