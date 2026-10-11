import torch, glob

for p in sorted(glob.glob('results/*/.pt', recursive=True)):
    try:
        s = torch.load(p, map_location='cpu')
        if 'item_emb.weight' in s:
            shape = tuple(s['item_emb.weight'].shape)
        else:
            shape = 'n/a'
        has_fair = 'fairness_alpha' in s
        print(f'{str(shape):<14} fair={has_fair:<5}  {p}')
    except Exception:
        print(f'{"ERR":<14}              {p}')