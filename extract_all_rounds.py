import json

grades = [3, 6, 9, 12]
for g in grades:
    path = f'evaluation_output/log_{g}_ELA_or_gemini-3-pro_matrix.json'
    with open(path) as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    rounds = data.get('rounds', [])
    
    print(f'=== Grade {g} ===')
    print(f'  Rounds: {len(rounds)}')
    print(f'  Best: {summary.get("best_pass_rate")}% @ round {summary.get("best_round")}')
    print(f'  Avg pass rate: {summary.get("average_pass_rate")}%')
    print(f'  Total time: {summary.get("total_elapsed")}')
    print(f'  Total cost: {summary.get("total_estimated_cost")}')
    
    for r in rounds:
        rnd = r.get('round', '?')
        s = r.get('summary', {})
        pr = s.get('pass_rate', '?')
        pc = s.get('pass_count', '?')
        nv = s.get('n_valid', '?')
        avg = s.get('breakdown', {}).get('by_type', {}).get('fill-in', {}).get('avg_score', '?')
        diff = s.get('breakdown', {}).get('by_difficulty', {})
        easy_pr = diff.get('easy', {}).get('pass_rate', '?')
        med_pr = diff.get('medium', {}).get('pass_rate', '?')
        hard_pr = diff.get('hard', {}).get('pass_rate', '?')
        print(f'  Round {rnd:>2}: {pr}% ({pc}/{nv})  avg={avg}  easy={easy_pr}% med={med_pr}% hard={hard_pr}%')
    print()
