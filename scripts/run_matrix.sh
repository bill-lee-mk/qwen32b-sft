#!/bin/bash
# ============================================================
# ELA 模型×年级 分数矩阵批量运行脚本
#
# 用法:
#   # 1. 单轮快速对比（冷启动，公平横向对比）
#   bash scripts/run_matrix.sh
#
#   # 2. 闭环改进 10 轮（每个组合跑 10 轮闭环，取最高分）
#   ROUNDS=10 bash scripts/run_matrix.sh
#
#   # 3. 只跑指定模型（并行时每个 tmux 窗口跑一个模型）
#   MODELS="fw/deepseek-r1" ROUNDS=10 bash scripts/run_matrix.sh
#
#   # 4. 自定义参数
#   MODELS="fw/deepseek-r1 fw/kimi-k2.5" GRADES="3 4 5" WORKERS=10 bash scripts/run_matrix.sh
#
#   # 5. 先试水后全量
#   ROUNDS=10 PILOT=50 bash scripts/run_matrix.sh
#
#   # 6. 只跑指定题型（默认 all = 同时生成 mcq+msq+fill-in）
#   QTYPE=mcq bash scripts/run_matrix.sh
#
#   # 7. 全部跑完后汇总矩阵
#   python scripts/summarize_matrix.py
#
# 默认行为:
#   每个 (模型,年级) 组合同时生成 MCQ + MSQ + FILL-IN 三种题型。
#
# 闭环流程（ROUNDS>1 时）:
#   每个 (模型,年级) 组合独立执行:
#     Round 1: 生成 → 评估 → 改进 prompt 规则
#     Round 2: 用改进后的 prompt 生成 → 评估 → 继续改进
#     ...
#     Round N: 最终生成 → 评估
#     → 自动保留历史最高分的题目和结果
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# === 可配置参数 ===
#MODELS="${MODELS:-fw/deepseek-r1 fw/deepseek-v3.2 fw/kimi-k2.5 fw/glm-5 fw/gpt-oss-120b fw/qwen3-235b}"
MODELS="${MODELS:-fw/deepseek-v3.2 fw/kimi-k2.5 fw/glm-5 fw/gpt-oss-120b}"
GRADES="${GRADES:-1 2 3 4 5 6 7 8 9 10 11 12}"
SUBJECT="${SUBJECT:-ELA}"
QTYPE="${QTYPE:-all}"           # 题型：all（默认，同时生成 mcq+msq+fill-in） / mcq / msq / fill-in
N="${N:-50}"                    # 单轮模式：--diverse N 采样题数
MODE="${MODE:-diverse}"         # 单轮模式：diverse / all
WORKERS="${WORKERS:-}"          # 并发线程数（空=按模型自动检测：kimi系20/其他Fireworks 50）
ROUNDS="${ROUNDS:-1}"           # 1=单轮快速对比，>1=闭环改进轮数
PILOT="${PILOT:-}"              # 闭环试水：先用小批量跑，最后全量（如 PILOT=50）
PASS_TARGET="${PASS_TARGET:-0}" # 闭环目标通过率（0=不设目标，跑满 ROUNDS 轮）
START_ROUND="${START_ROUND:-1}" # 从第几轮开始（默认 1）；续跑时设为已完成轮数+1
OUTDIR="${OUTDIR:-evaluation_output/matrix}"
RUN_ID="${RUN_ID:-matrix}"      # 闭环 run-id，隔离不同批次的 examples/rules

mkdir -p "$OUTDIR"

echo "============================================================"
echo "  矩阵批量运行配置"
echo "  模型: $MODELS"
echo "  年级: $GRADES"
echo "  学科: $SUBJECT"
echo "  题型: $QTYPE"
if [ "$ROUNDS" -gt 1 ]; then
  echo "  策略: 闭环改进 $ROUNDS 轮（生成→评估→补范例→改prompt→重复）"
  if [ -n "$PILOT" ]; then
    echo "  试水: 每轮先生成 $PILOT 题试水，最后全量"
  fi
  echo "  目标: pass_rate >= ${PASS_TARGET}%（0=不设目标）"
else
  if [ "$MODE" = "all" ]; then
    echo "  策略: 单轮全量组合 (--all-combinations)"
  else
    echo "  策略: 单轮采样 N=$N 题/年级 (--diverse $N)"
  fi
fi
echo "  并发: ${WORKERS:-自动检测}"
echo "  输出: $OUTDIR/"
echo "============================================================"

TOTAL=0
OK=0
FAIL=0
START_TIME=$(date +%s)

for MODEL in $MODELS; do
  MNAME=$(echo "$MODEL" | tr '/' '_' | tr '.' '_')
  for GRADE in $GRADES; do
    TOTAL=$((TOTAL + 1))
    TAG="${MNAME}_${GRADE}_${SUBJECT}"

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  [$TOTAL] Model=$MODEL  Grade=$GRADE  Subject=$SUBJECT  Type=$QTYPE  Rounds=$ROUNDS"
    echo "────────────────────────────────────────────────────────────"

    if [ "$ROUNDS" -gt 1 ]; then
      # ========================================
      # 闭环改进模式：main.py closed-loop
      # ========================================
      LOOP_ARGS="--model $MODEL --grade $GRADE --subject $SUBJECT --type $QTYPE"
      LOOP_ARGS="$LOOP_ARGS --max-rounds $ROUNDS --start-round $START_ROUND --pass-rate-target $PASS_TARGET"
      LOOP_ARGS="$LOOP_ARGS --run-id $RUN_ID"
      if [ -n "$PILOT" ]; then
        LOOP_ARGS="$LOOP_ARGS --pilot-batch $PILOT"
      fi
      if [ -n "$WORKERS" ]; then
        LOOP_ARGS="$LOOP_ARGS --workers $WORKERS"
      fi
      echo "  执行: python main.py closed-loop $LOOP_ARGS"

      if python main.py closed-loop $LOOP_ARGS; then
        echo "  [闭环完成] Model=$MODEL Grade=$GRADE"
      else
        echo "  [闭环失败] Model=$MODEL Grade=$GRADE"
        FAIL=$((FAIL + 1))
        continue
      fi

      # 闭环完成后，找到 best 结果文件并复制到 matrix 目录
      SCOPE_TAG="${GRADE}_${SUBJECT}"
      BEST_PATTERN="evaluation_output/results_${SCOPE_TAG}_${MNAME}_${RUN_ID}_best_*.json"
      BEST_FILE=$(ls $BEST_PATTERN 2>/dev/null | head -1 || true)

      if [ -n "$BEST_FILE" ] && [ -f "$BEST_FILE" ]; then
        cp "$BEST_FILE" "$OUTDIR/results_${TAG}.json"
        BEST_MCQS=$(echo "$BEST_FILE" | sed 's/results_/mcqs_/')
        [ -f "$BEST_MCQS" ] && cp "$BEST_MCQS" "$OUTDIR/mcqs_${TAG}.json"
        RATE=$(python3 -c "import json; print(json.load(open('$BEST_FILE')).get('pass_rate', 0))")
        echo "  [最佳结果] pass_rate=${RATE}% → $OUTDIR/results_${TAG}.json"
        OK=$((OK + 1))
      else
        FALLBACK="evaluation_output/results_${SCOPE_TAG}_${MNAME}_${RUN_ID}_round1.json"
        if [ -f "$FALLBACK" ]; then
          cp "$FALLBACK" "$OUTDIR/results_${TAG}.json"
          RATE=$(python3 -c "import json; print(json.load(open('$FALLBACK')).get('pass_rate', 0))")
          echo "  [结果] pass_rate=${RATE}% → $OUTDIR/results_${TAG}.json"
          OK=$((OK + 1))
        else
          echo "  [警告] 未找到结果文件: $BEST_PATTERN"
          FAIL=$((FAIL + 1))
        fi
      fi

    else
      # ========================================
      # 单轮快速模式：generate + evaluate
      # ========================================
      MCQS="$OUTDIR/mcqs_${TAG}.json"
      RESULT="$OUTDIR/results_${TAG}.json"

      GEN_ARGS="--model $MODEL --grade $GRADE --subject $SUBJECT --type $QTYPE --output $MCQS"
      if [ -n "$WORKERS" ]; then
        GEN_ARGS="$GEN_ARGS --workers $WORKERS"
      fi
      if [ "$MODE" = "all" ]; then
        GEN_ARGS="$GEN_ARGS --all-combinations"
      else
        GEN_ARGS="$GEN_ARGS --diverse $N"
      fi

      if python scripts/generate_questions.py $GEN_ARGS; then
        echo "  [生成完成] $MCQS"
      else
        echo "  [生成失败] Model=$MODEL Grade=$GRADE"
        FAIL=$((FAIL + 1))
        continue
      fi

      if python main.py evaluate --input "$MCQS" --output "$RESULT" --parallel 20; then
        echo "  [评估完成] $RESULT"
        OK=$((OK + 1))
      else
        echo "  [评估失败] Model=$MODEL Grade=$GRADE"
        FAIL=$((FAIL + 1))
      fi
    fi
  done
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_R=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  批量运行完成"
echo "  成功=$OK / 失败=$FAIL / 总计=$TOTAL"
echo "  总耗时: ${MINUTES}m ${SECONDS_R}s"
echo "  结果目录: $OUTDIR/"
echo ""
echo "  查看分数矩阵:"
echo "    python scripts/summarize_matrix.py --dir $OUTDIR"
echo "============================================================"
