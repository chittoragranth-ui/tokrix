"""Compare synthetic routing quality, normalized cost, and latency offline."""

from tokrix.routing import (
    evaluate_routing_performance,
    get_performance_benchmark_dataset,
)


def main() -> None:
    dataset = get_performance_benchmark_dataset()
    result = evaluate_routing_performance(dataset)

    print("Tokrix synthetic routing performance benchmark")
    print(f"Evaluated scenarios: {result.case_count}")
    print("These deterministic assumptions are not provider measurements.")
    print()
    print(f"{'Metric':<22}{'Static':>14}{'Adaptive':>14}{'Difference':>16}")
    for label, static, adaptive, unit in (
        (
            "Average quality",
            result.static_metrics.average_quality_score,
            result.adaptive_metrics.average_quality_score,
            "points",
        ),
        (
            "Average cost",
            result.static_metrics.average_cost_units,
            result.adaptive_metrics.average_cost_units,
            "units",
        ),
        (
            "Average latency",
            result.static_metrics.average_latency_ms,
            result.adaptive_metrics.average_latency_ms,
            "ms",
        ),
    ):
        print(
            f"{label:<22}{static:>14.2f}{adaptive:>14.2f}"
            f"{adaptive - static:>+12.2f} {unit}"
        )

    print()
    print("Per-scenario selected tiers and measurements are available in")
    print("result.case_results; aggregate differences are adaptive minus static.")


if __name__ == "__main__":
    main()
