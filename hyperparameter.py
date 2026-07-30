import pandas as pd
import ast

def hyperparam_dispersion(csv_path, baseline):
    df = pd.read_csv(csv_path)
    rows = df[(df["baseline"] == baseline) & (df["params"].notna()) & (df["params"] != "")]
    rows = rows.drop_duplicates("fold")  # one params dict per fold
    parsed = rows["params"].apply(ast.literal_eval)
    param_df = pd.DataFrame(list(parsed), index=rows["fold"])
    print(f"=== {baseline}: tuned hyperparameters by fold ===")
    print(param_df)
    print()
    print("dispersion (coefficient of variation, std/mean, per numeric param):")
    numeric = param_df.select_dtypes(include="number")
    print((numeric.std() / numeric.mean().abs()).round(2))
    return param_df

hyperparam_dispersion("outputs/results_5fold.csv", "LSTM")