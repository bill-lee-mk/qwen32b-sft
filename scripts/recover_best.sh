#!/bin/bash
# 恢复脚本：对已有的 mcqs_round2 文件跑评估，生成 best 文件
# 用法：bash scripts/recover_best.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PARALLEL=20

MCQS_FILES=(
  "evaluation_output/mcqs_1_ELA_or_gpt-5_2_matrix_round2.json"
  "evaluation_output/mcqs_7_ELA_or_gpt-5_2_matrix_round2.json"
  "evaluation_output/mcqs_1_ELA_or_gemini-3-pro_matrix_round2.json"
  "evaluation_output/mcqs_7_ELA_or_gemini-3-pro_matrix_round2.json"
)

for MCQS in "${MCQS_FILES[@]}"; do
  if [ ! -f "$MCQS" ]; then
    echo "[跳过] 文件不存在: $MCQS"
    continue
  fi

  RESULTS="${MCQS/mcqs_/results_}"
  echo ""
  echo "========================================"
  echo "  评估: $(basename "$MCQS")"
  echo "========================================"

  if python main.py evaluate --input "$MCQS" --output "$RESULTS" --parallel "$PARALLEL"; then
    echo "[评估完成] $RESULTS"

    # 从 results 中读取 pass_rate，创建 best 文件
    python3 -c "
import json, shutil, os
results_path = '$RESULTS'
mcqs_path = '$MCQS'
with open(results_path) as f:
    data = json.load(f)
pr = data.get('pass_rate', 0)
rate_str = str(round(pr, 1)).replace('.', '_')
base_results = results_path.replace('_round2.json', '')
base_mcqs = mcqs_path.replace('_round2.json', '')
best_r = f'{base_results}_best_{rate_str}.json'
best_m = f'{base_mcqs}_best_{rate_str}.json'
shutil.copy2(results_path, best_r)
shutil.copy2(mcqs_path, best_m)
os.remove(results_path)
os.remove(mcqs_path)
print(f'  [已保存] {os.path.basename(best_r)}  pass_rate={pr}%')
print(f'  [已保存] {os.path.basename(best_m)}')
"
  else
    echo "[评估失败] $MCQS"
  fi
done

echo ""
echo "========================================"
echo "  恢复完成，运行汇总查看结果:"
echo "  python scripts/summarize_matrix.py --dir evaluation_output"
echo "========================================"
