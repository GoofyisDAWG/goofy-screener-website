"""
Smoke test — confirms known features are actually present in app.py.

Exists because local cron and interactive git pushes have twice this week
collided into a conflicted rebase that got resolved with a hard reset,
silently discarding real feature commits (the regime indicator got wiped
this way once already). A missing UI element is easy to not notice for
days; this catches it in one second.

Run manually any time: python3 verify_features.py
Exits non-zero (and prints exactly what's missing) if anything regresses.
"""
import sys

REQUIRED = {
    "Market regime indicator (Home page)":
        ["REGIME_ALLOWED_STRATEGIES", "regime_title", "Market Conditions Right Now"],
    "Compounded, position-sized equity curves (Home/Track Record)":
        ['(1 + (_bdf["pnl_pct"] / 100) * _bsz).cumprod()',
         '(1 + (_rdf["pnl_pct"] / 100) * _rsz).cumprod()',
         '(1 + (dc_sorted["pnl_pct"] / 100) * _dsz).cumprod()'],
    "size_pct actually loaded into trade history":
        ['"size_pct":    t.get("size_pct")'],
    "Investment Simulator position-fraction compounding (not full-balance)":
        ["_sim_position_pct", "_s_val *= (1 + _sim_position_pct",
         "_sim_val *= (1 + _sim_position_pct"],
    "Moomoo broker option":
        ['"Moomoo ($3 AUD or 0.03%, whichever is greater)"'],
}

def main():
    with open("app.py", encoding="utf-8") as f:
        src = f.read()

    missing = {}
    for feature, markers in REQUIRED.items():
        gone = [m for m in markers if m not in src]
        if gone:
            missing[feature] = gone

    if not missing:
        print(f"✅ All {len(REQUIRED)} tracked features present in app.py.")
        return 0

    print(f"❌ {len(missing)} feature(s) missing or partially missing:\n")
    for feature, gone in missing.items():
        print(f"  {feature}")
        for m in gone:
            print(f"    missing marker: {m!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
