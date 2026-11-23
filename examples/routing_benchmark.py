"""Run Tokrix's deterministic, offline routing benchmark."""

from tokrix.routing import (
    ModelTier,
    compare_routing_results,
    evaluate_routing,
    get_benchmark_dataset,
)


def main() -> None:
    dataset = get_benchmark_dataset()
    cases = [entry.benchmark_case for entry in dataset]
    result = evaluate_routing(cases)
    comparison = compare_routing_results(result)

    print("Tokrix offline routing benchmark")
    print(f"Evaluated cases: {result.case_count}")
    print()
    print("Verification success rate")
    print(f"  Static:   {result.static_metrics.success_rate:.1%}")
    print(f"  Adaptive: {result.adaptive_metrics.success_rate:.1%}")
    print(
        "  Difference (adaptive - static): "
        f"{comparison.success_rate_difference_percentage_points:+.2f} pp"
    )
    print()
    print("Average tier-cost proxy")
    print(
        "  Static:   "
        f"{result.static_metrics.average_tier_cost_proxy:.2f}"
    )
    print(
        "  Adaptive: "
        f"{result.adaptive_metrics.average_tier_cost_proxy:.2f}"
    )
    print(
        "  Difference (adaptive - static): "
        f"{comparison.average_tier_cost_proxy_difference:+.2f}"
    )

    relative_change = comparison.relative_tier_cost_proxy_change_percent
    if relative_change is None:
        print("  Relative change: N/A (not comparable)")
    else:
        direction = {
            "more_expensive": "more expensive",
            "less_expensive": "less expensive",
            "same_cost": "same cost",
        }[comparison.adaptive_cost_direction]
        print(f"  Relative change: {relative_change:+.2f}% ({direction})")

    print()
    print("Tier distribution difference (adaptive - static)")
    for tier in ModelTier:
        difference = comparison.tier_distribution_differences[tier]
        print(f"  {tier.value.upper():<6} {difference:+d} cases")

    outcomes = comparison.outcome_comparison
    print()
    print("Adaptive verification outcomes vs. static")
    for label, count, percentage in (
        ("Improved", outcomes.improved_count, outcomes.improved_percentage),
        ("Degraded", outcomes.degraded_count, outcomes.degraded_percentage),
        ("Unchanged", outcomes.unchanged_count, outcomes.unchanged_percentage),
    ):
        percentage_text = f"{percentage:.1f}%" if percentage is not None else "N/A"
        print(f"  {label:<9} {count:>3} ({percentage_text})")


if __name__ == "__main__":
    main()
