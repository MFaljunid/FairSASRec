import pandas as pd
from collections import defaultdict

for name in ['ml-100k-fairir', 'ml-1m-fairir']:
    try:
        tr = pd.read_csv(f'data/{name}/train.csv')
        lens = tr.groupby('user_id').size()
        for w in [20, 50, 100]:
            cov = (lens.clip(upper=w) / lens).mean()
            print(f"{name:16} window={w:3}  "
                  f"mean_hist={lens.mean():.0f}  "
                  f"coverage={cov:.0%}")
    except FileNotFoundError:
        print(f"{name}: not on this machine")
    print()