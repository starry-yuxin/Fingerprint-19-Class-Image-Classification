import argparse
import numpy as np
import pandas as pd

EPS = 1e-12

def probs_to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return np.log(p)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=1, keepdims=True), EPS, None)

def parse_bias_str(s: str, C: int) -> np.ndarray:
    b = np.zeros(C, dtype=np.float32)
    if not s:
        return b
    items = [t.strip() for t in s.split(",") if t.strip()]
    for it in items:
        k, v = it.split("=")
        b[int(k)] = float(v)
    return b

def parse_seed_weights(s: str):
    # "0.56,0.04,0.40"
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--seed_weights must be like '0.56,0.04,0.40'")
    w = [float(x) for x in parts]
    s = sum(w)
    if s <= 0:
        raise ValueError("seed_weights sum must be > 0")
    return [w[0]/s, w[1]/s, w[2]/s]

def fuse_logits(L_list, w_list, bias):
    z = np.zeros_like(L_list[0], dtype=np.float32)
    for L, w in zip(L_list, w_list):
        z += float(w) * L
    z += bias[None, :]
    return z

def acc_from_logits(z, y):
    pred = z.argmax(1)
    return float((pred == y).mean())

def simplex_grid_3(step: float):
    n = int(round(1.0 / step))
    for ia in range(n + 1):
        wa = ia * step
        for ib in range(n + 1 - ia):
            wb = ib * step
            wc = 1.0 - wa - wb
            if wc < -1e-9:
                continue
            if wc < 0:
                wc = 0.0
            yield float(wa), float(wb), float(wc)

def greedy_bias_tune(LA, LB, LC, w_best, y, init_bias, bias_min, bias_max, bias_step, passes):
    K = init_bias.shape[0]
    bias = init_bias.copy()

    grid = np.arange(bias_min, bias_max + 1e-9, bias_step, dtype=np.float32)

    z0 = fuse_logits([LA, LB, LC], w_best, bias=bias)
    best_acc = acc_from_logits(z0, y)

    for ps in range(passes):
        improved_any = False
        for k in range(K):
            cur = float(bias[k])
            best_k = cur
            best_k_acc = best_acc

            for v in grid:
                v = float(v)
                if abs(v - cur) < 1e-12:
                    continue
                bias[k] = v
                z_try = fuse_logits([LA, LB, LC], w_best, bias=bias)
                acc = acc_from_logits(z_try, y)
                if acc > best_k_acc + 1e-12:
                    best_k_acc = acc
                    best_k = v

            bias[k] = cur  # restore

            if best_k_acc > best_acc + 1e-12:
                bias[k] = best_k
                best_acc = best_k_acc
                improved_any = True

        if not improved_any:
            break

    return best_acc, bias

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_a", required=True)
    ap.add_argument("--pred_b", required=True)
    ap.add_argument("--pred_c", required=True)

    ap.add_argument("--name_a", default="A")
    ap.add_argument("--name_b", default="B")
    ap.add_argument("--name_c", default="C")

    ap.add_argument("--do_sweep", action="store_true")
    ap.add_argument("--step", type=float, default=0.01)

    ap.add_argument("--w_a", type=float, default=0.56)
    ap.add_argument("--w_b", type=float, default=0.04)
    ap.add_argument("--w_c", type=float, default=0.40)

    ap.add_argument("--seed_weights", type=str, default="",
                    help='Optional candidate weights, e.g. "0.56,0.04,0.40". '
                         "If provided, it will be evaluated as a candidate too.")

    ap.add_argument("--do_bias", action="store_true")
    ap.add_argument("--init_bias", type=str, default="4=0.15,7=0.30,10=0.15")
    ap.add_argument("--bias_min", type=float, default=-0.30)
    ap.add_argument("--bias_max", type=float, default=0.30)
    ap.add_argument("--bias_step", type=float, default=0.01)
    ap.add_argument("--passes", type=int, default=6)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_params", default="")
    args = ap.parse_args()

    A = pd.read_csv(args.pred_a)
    B = pd.read_csv(args.pred_b)
    C = pd.read_csv(args.pred_c)

    # align
    A = A.sort_values("image_path").reset_index(drop=True)
    B = B.sort_values("image_path").reset_index(drop=True)
    C = C.sort_values("image_path").reset_index(drop=True)
    assert (A.image_path.values == B.image_path.values).all()
    assert (A.image_path.values == C.image_path.values).all()

    prob_cols = [c for c in A.columns if c.startswith("prob_")]
    K = len(prob_cols)
    print(f"Loaded N={len(A)}, C={K}")

    if "true_class_id" not in A.columns:
        raise RuntimeError("Need true_class_id in CSVs to evaluate acc (you already have it).")
    y = A["true_class_id"].values.astype(int)

    pA = A[prob_cols].values.astype(np.float32)
    pB = B[prob_cols].values.astype(np.float32)
    pC = C[prob_cols].values.astype(np.float32)

    LA = probs_to_logits(pA)
    LB = probs_to_logits(pB)
    LC = probs_to_logits(pC)

    # build candidate weight list
    candidates = []

    # fixed weights candidate
    s = float(args.w_a + args.w_b + args.w_c)
    if s <= 0:
        raise RuntimeError("Sum of fixed weights must be > 0")
    w_fixed = [args.w_a / s, args.w_b / s, args.w_c / s]
    candidates.append(("fixed", w_fixed))

    # seed candidate (optional)
    if args.seed_weights.strip():
        w_seed = parse_seed_weights(args.seed_weights)
        candidates.append(("seed", w_seed))

    # sweep-best candidate (optional)
    if args.do_sweep:
        best_acc = -1.0
        best_w = None
        total = 0
        print(f"Sweeping weights on simplex with step={args.step} ...")
        for wa, wb, wc in simplex_grid_3(args.step):
            z = fuse_logits([LA, LB, LC], [wa, wb, wc], bias=np.zeros(K, dtype=np.float32))
            acc = acc_from_logits(z, y)
            total += 1
            if acc > best_acc + 1e-12:
                best_acc = acc
                best_w = [wa, wb, wc]
        print("\n=== BEST (sweep, NO bias) ===")
        print(f"acc = {best_acc:.4f}")
        print(f"weights: {args.name_a}={best_w[0]:.3f}, {args.name_b}={best_w[1]:.3f}, {args.name_c}={best_w[2]:.3f}")
        print(f"searched combos = {total}")
        candidates.append(("sweep_best", best_w))

    # Evaluate candidates
    init_bias = parse_bias_str(args.init_bias, K)

    results = []
    for tag, w in candidates:
        # no-bias acc
        z_nb = fuse_logits([LA, LB, LC], w, bias=np.zeros(K, dtype=np.float32))
        acc_nb = acc_from_logits(z_nb, y)

        if args.do_bias:
            acc_final, bias_final = greedy_bias_tune(
                LA, LB, LC, w, y,
                init_bias=init_bias,
                bias_min=args.bias_min,
                bias_max=args.bias_max,
                bias_step=args.bias_step,
                passes=args.passes
            )
        else:
            acc_final, bias_final = acc_nb, np.zeros(K, dtype=np.float32)

        results.append((tag, w, acc_nb, acc_final, bias_final))

    # pick best by FINAL acc (this is the key fix)
    results.sort(key=lambda x: x[3], reverse=True)
    best_tag, best_w, best_acc_nb, best_acc, best_bias = results[0]

    print("\n=== CANDIDATES SUMMARY ===")
    for tag, w, acc_nb, acc_f, _ in results:
        print(f"{tag:10s}  no_bias={acc_nb:.4f}  final={acc_f:.4f}  w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]")
    print("\n=== PICK BEST (by FINAL acc) ===")
    print(f"best_tag = {best_tag}")
    print(f"best final acc = {best_acc:.4f}")
    print(f"best weights: {args.name_a}={best_w[0]:.3f}, {args.name_b}={best_w[1]:.3f}, {args.name_c}={best_w[2]:.3f}")

    # Export final fused probs/preds
    z_final = fuse_logits([LA, LB, LC], best_w, bias=best_bias)
    probs_final = softmax(z_final)
    pred = probs_final.argmax(1)

    out = pd.DataFrame({
        "image_path": A["image_path"].values,
        "true_class_id": y,
        "pred_class_id": pred.astype(int),
    })
    for j in range(K):
        out[f"prob_{j}"] = probs_final[:, j]

    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("Saved:", args.out_csv)

    if args.out_params:
        with open(args.out_params, "w", encoding="utf-8") as f:
            f.write("picked_candidate:\n")
            f.write(f"{best_tag}\n\n")
            f.write("best_weights:\n")
            f.write(f"{args.name_a}={best_w[0]:.6f}\n")
            f.write(f"{args.name_b}={best_w[1]:.6f}\n")
            f.write(f"{args.name_c}={best_w[2]:.6f}\n")
            f.write("\nnonzero_bias:\n")
            for i in range(K):
                if abs(float(best_bias[i])) > 1e-9:
                    f.write(f"{i}={float(best_bias[i]):+.6f}\n")
        print("Saved params:", args.out_params)

if __name__ == "__main__":
    main()
