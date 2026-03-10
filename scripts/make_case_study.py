import argparse
import os
import pandas as pd


def zscore(series):
    std = series.std()
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="./results/raw_metrics.csv")
    parser.add_argument("--output_csv", type=str, default="./results/case_study.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    df = pd.read_csv(args.input_csv).copy()

    # 标准化后看“分歧”
    df["psnr_z"] = zscore(df["psnr"])
    df["qalign_z"] = zscore(df["qalign_score"])
    df["lpips_z"] = zscore(-df["lpips"])  # 越大越好，取负号转成同向

    # Case A: PSNR高但Q-Align低
    df["caseA_score"] = df["psnr_z"] - df["qalign_z"]

    # Case B: PSNR低但Q-Align高
    df["caseB_score"] = df["qalign_z"] - df["psnr_z"]

    # Case C: LPIPS好但Q-Align低
    df["caseC_score"] = df["lpips_z"] - df["qalign_z"]

    # Case D: LPIPS差但Q-Align高
    df["caseD_score"] = df["qalign_z"] - df["lpips_z"]

    case_rows = []

    for case_name, col in [
        ("PSNR_high_QAlign_low", "caseA_score"),
        ("PSNR_low_QAlign_high", "caseB_score"),
        ("LPIPS_good_QAlign_low", "caseC_score"),
        ("LPIPS_bad_QAlign_high", "caseD_score"),
    ]:
        topk = df.sort_values(col, ascending=False).head(3).copy()
        topk["case_type"] = case_name
        case_rows.append(topk)

    out_df = pd.concat(case_rows, ignore_index=True)
    keep_cols = [
        "case_type", "image_name", "base_name", "degradation",
        "psnr", "ssim", "lpips", "qalign_score",
        "gt_path", "pred_path"
    ]
    out_df = out_df[keep_cols]
    out_df.to_csv(args.output_csv, index=False)

    print("Saved case study to:", args.output_csv)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
