import json

grades = [3, 6, 9, 12]
for g in grades:
    path = f'evaluation_output/closed_loop_progress_{g}_ELA_or_gemini-3-pro_matrix.json'
    with open(path) as f:
        data = json.load(f)
    rounds = data.get('rounds', [])
    best_pr = data.get('best_pass_rate', '?')
    best_rnd = data.get('best_round', '?')
    cost = data.get('total_cost_usd', 0)
    time_min = data.get('total_time_min', 0)
    print(f'=== Grade {g} ===')
    print(f'Total rounds: {len(rounds)}')
    print(f'Best: {best_pr}% @ round {best_rnd}')
    print(f'Total cost: ${cost:.2f}')
    print(f'Total time: {time_min:.1f} min')
    for r in rounds:
        rnd = r.get('round', '?')
        pr = r.get('pass_rate', r.get('eval_pass_rate', '?'))
        print(f'  Round {rnd}: {pr}%')
    print()
