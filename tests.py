import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.agent.pipeline import parse_user_input, execute_plan
from src.config import settings


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int((len(arr) - 1) * q)
    return arr[idx]


def ensure_fixture_files() -> None:
    bench_dir = settings.OUTPUT_DIR / "bench"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "clan.txt").write_text(
        "A clan is a social group sharing common ancestry and traditions. "
        "Historically, clans organized governance, kinship, and protection.",
        encoding="utf-8",
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases.json"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Also run execute_plan (has side effects).")
    args = parser.parse_args()

    ensure_fixture_files()
    cases = load_cases(args.cases)

    # Lightweight synthetic history for context-sensitive prompts
    history = [
        HumanMessage(content="my name is harsh"),
    ]

    # Warmup
    for _ in range(args.warmup):
        for c in cases:
            _ = parse_user_input(c["prompt"], history)

    rows = []
    all_parse = []
    all_e2e = []
    correct = 0
    total = 0

    for c in cases:
        parse_times = []
        e2e_times = []
        expected = c.get("expected_first_intent")

        for _ in range(args.runs):
            t0 = time.perf_counter()
            plan = parse_user_input(c["prompt"], history)
            t1 = time.perf_counter()
            parse_times.append((t1 - t0) * 1000.0)

            if args.execute:
                t2 = time.perf_counter()
                _ = execute_plan(plan)
                t3 = time.perf_counter()
                e2e_times.append((t3 - t0) * 1000.0)

        first_intent = (plan[0].get("intent") if plan else "none")
        if expected:
            total += 1
            if first_intent == expected:
                correct += 1

        all_parse.extend(parse_times)
        all_e2e.extend(e2e_times)

        rows.append({
            "id": c["id"],
            "expected": expected,
            "actual": first_intent,
            "parse_avg_ms": round(statistics.mean(parse_times), 2),
            "parse_p95_ms": round(pctl(parse_times, 0.95), 2),
            "e2e_avg_ms": round(statistics.mean(e2e_times), 2) if e2e_times else None,
        })

    print("\n=== Case Results ===")
    for r in rows:
        print(
            f'- {r["id"]}: expected={r["expected"]}, actual={r["actual"]}, '
            f'parse_avg={r["parse_avg_ms"]}ms, parse_p95={r["parse_p95_ms"]}ms'
            + (f', e2e_avg={r["e2e_avg_ms"]}ms' if r["e2e_avg_ms"] is not None else "")
        )

    print("\n=== Aggregate ===")
    print(f"Model(router): {settings.ROUTER_LLM}")
    print(f"Cases: {len(cases)}, Runs/case: {args.runs}, Warmup: {args.warmup}")
    print(f"Parse avg: {round(statistics.mean(all_parse), 2)} ms")
    print(f"Parse p50: {round(pctl(all_parse, 0.50), 2)} ms")
    print(f"Parse p95: {round(pctl(all_parse, 0.95), 2)} ms")
    if all_e2e:
        print(f"E2E avg: {round(statistics.mean(all_e2e), 2)} ms")
        print(f"E2E p50: {round(pctl(all_e2e, 0.50), 2)} ms")
        print(f"E2E p95: {round(pctl(all_e2e, 0.95), 2)} ms")
    if total > 0:
        print(f"First-intent accuracy: {correct}/{total} = {round((correct/total)*100, 1)}%")


if __name__ == "__main__":
    main()