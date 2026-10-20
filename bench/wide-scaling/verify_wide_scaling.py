"""Recompute the 3,000-call option-scaling tables from the included observations.

Checks row coverage (every observation present exactly once), probability validity, every
accuracy count in summary.json, and the paired selection totals and confidence tables
recomputed from this package's own per-row outputs. No network access; no external data.
"""
from math import isfinite, log
from pathlib import Path
import argparse, json

HERE = Path(__file__).resolve().parent
KEY_ORDER = ["16", "32", "64", "128", "256"]


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def ece(pairs, bins=10):
    b = [[] for _ in range(bins)]
    for c, y in pairs:
        b[min(bins - 1, int(c * bins))].append((c, y))
    total = len(pairs); e = 0.0
    for chunk in b:
        if chunk:
            conf = sum(x for x, _ in chunk) / len(chunk); acc = sum(y for _, y in chunk) / len(chunk)
            e += len(chunk) / total * abs(acc - conf)
    return e


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    obs = read_jsonl(HERE / "observations.jsonl")
    conf_sum = summary["confidence_valid_rows_only_denominators_disclosed"]
    stab_sum = summary["option_order_stability"]
    acc_sum = summary["accuracy_by_k"]
    paired_sum = summary["paired_scaling"]
    assert len(obs) == 3000 and summary["measured_calls"] == 3000
    seen = set(); by_k = {}
    for o in obs:
        ident = (o["repeat"], o["n_options"], o["order_index"], o["item_uid"])
        assert ident not in seen, ident
        seen.add(ident)
        keys = o["option_keys"]
        assert len(keys) == o["n_options"] and len(set(keys)) == len(keys)
        p = o["probabilities"]
        if o["valid"]:
            assert p is not None and len(p) == o["n_options"]
            assert all(isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p) - 1) < 1e-6
            assert keys[max(range(len(p)), key=lambda i: p[i])] == o["selected"]
        else:
            assert p is None and o["selected"] is None and o["correct"] is False
        by_k.setdefault(str(o["n_options"]), []).append(o)
    assert sorted(by_k, key=int) == KEY_ORDER and all(len(v) == 600 for v in by_k.values())
    report = {"observations": len(obs), "errors": sum(not o["valid"] for o in obs),
              "accuracy_counts_match": True, "confidence_tables_match": True,
              "paired_selection_totals_match": True, "order_stability_totals_match": True,
              "accuracy_by_k_primary_correct": {}}
    for k, rows in by_k.items():
        prim = [x for x in rows if x["order_index"] == 0]; ctrl = [x for x in rows if x["order_index"] == 1]
        assert len(prim) == 300 and len(ctrl) == 300, (k, len(prim), len(ctrl))
        tab = acc_sum[k]
        assert tab["primary_correct"] == sum(x["correct"] for x in prim), (k, "primary")
        report["accuracy_by_k_primary_correct"][k] = tab["primary_correct"]
        assert tab["shuffled_correct"] == sum(x["correct"] for x in ctrl), (k, "shuffled")
        assert tab["invalid"] == sum(not x["valid"] for x in rows), (k, "invalid")
        assert tab["exact_ties"] == sum(int(x["exact_tie"]) for x in rows if x["valid"]), (k, "ties")
        # confidence tables cover ALL 600 rows at this K (both orderings), valid rows only
        allv = [x for x in rows if x["valid"]]
        pmax = [max(x["probabilities"]) for x in allv]
        assert len(allv) == conf_sum[k]["valid_n"] == conf_sum[k]["all_n"], (k, "n")
        assert approx(ece(list(zip(pmax, [x["correct"] for x in allv]))), conf_sum[k]["ece_pmax_10bin"], 1e-12), (k, "ece")
        assert approx(sum(pmax) / len(pmax), conf_sum[k]["mean_pmax"], 1e-12), (k, "mean_pmax")
        assert approx(sum(x["correct"] for x in allv) / len(allv), conf_sum[k]["accuracy_valid"], 1e-12), (k, "acc_valid")
        nll = [-log(max(x["p_gold"], 1e-300)) for x in allv]
        assert approx(sum(nll) / len(nll), conf_sum[k]["nll_mean"], 1e-9), (k, "nll")
        brier = []
        for x in allv:
            g = x["option_keys"].index(summary["gold_keys_by_observation"][x["observation_id"]])
            brier.append(sum((pp - (1.0 if i == g else 0.0)) ** 2 for i, pp in enumerate(x["probabilities"])))
        assert approx(sum(brier) / len(brier), conf_sum[k]["brier_mean"], 1e-9), (k, "brier")
        a = {(x["repeat"], x["item_uid"]): x for x in rows if x["order_index"] == 0}
        b2 = {(x["repeat"], x["item_uid"]): x for x in rows if x["order_index"] == 1}
        assert set(a) == set(b2) and len(a) == 300
        valid_pairs = [x for x in a if a[x]["valid"] and b2[x]["valid"]]
        assert len(valid_pairs) == stab_sum[k]["both_valid"] == stab_sum[k]["pairs"], (k, "pairs")
        assert stab_sum[k]["invalid_pairs"] == 300 - len(valid_pairs)
        shifts = [abs(a[x]["p_gold"] - b2[x]["p_gold"]) for x in valid_pairs]
        assert approx(sum(shifts) / len(shifts), stab_sum[k]["mean_abs_p_gold_shift"], 1e-9), (k, "shift")
    for label in ("K256_vs_K16", "K32_vs_K16", "K64_vs_K32", "K128_vs_K64", "K256_vs_K128"):
        v = paired_sum[label]
        kb = int(label[1:label.index("_vs_")]); ka = int(label.split("_vs_")[1][1:])
        a = {(x["repeat"], x["item_uid"]): x for x in by_k[str(ka)] if x["order_index"] == 0}
        b3 = {(x["repeat"], x["item_uid"]): x for x in by_k[str(kb)] if x["order_index"] == 0}
        assert set(a) == set(b3) and len(a) == 300
        valid_pairs = [x for x in a if a[x]["valid"] and b3[x]["valid"]]
        both = sum(1 for x in valid_pairs if a[x]["correct"] and b3[x]["correct"])
        only_a = sum(1 for x in valid_pairs if a[x]["correct"] and not b3[x]["correct"])
        only_b = sum(1 for x in valid_pairs if b3[x]["correct"] and not a[x]["correct"])
        assert v["both"] == both and v["only_K%d" % ka] == only_a and v["only_K%d" % kb] == only_b, (label, v, both, only_a, only_b)
        assert approx(v["diff_acc_kb_minus_ka"], (only_b - only_a) / 300, 1e-12), label
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    main()
