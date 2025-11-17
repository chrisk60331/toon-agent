from __future__ import annotations

from pathlib import Path

from benchmark import benchmark_with_score

SUMMARY_PATH = Path(__file__).resolve().parent / "prize_summary.txt"


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Reference summary file not found at {SUMMARY_PATH}. Run documentation to regenerate."
        )

    reference_text = SUMMARY_PATH.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise ValueError(f"Reference summary at {SUMMARY_PATH} is empty.")

    results = benchmark_with_score(reference_text)
    # Sort by F1 descending, then duration ascending for readability.
    results.sort(key=lambda item: (-item.f1_score, item.duration_seconds))

    print(f"Reference summary loaded from: {SUMMARY_PATH}")
    for result in results:
        total_tokens = result.summary.total_tokens
        print(
            f"{result.name}: f1={result.f1_score:.3f} | duration={result.duration_seconds:.3f}s | "
            f"total_tokens={total_tokens if total_tokens is not None else 'n/a'}"
        )


if __name__ == "__main__":
    main()

