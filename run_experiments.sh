#!/bin/bash
# =============================================================================
# FedMIA 消融实验启动脚本
#
# 实验设置：
#   - 数据集：CIFAR-100 预提取特征 (data/cifar100_features/)
#   - 客户端数：10，每轮全部参与 (sample_client_num=10)
#   - 训练轮次：300
#   - 攻击方法：loss_series / avg_cosine / fedmia_i / fedmia_ii
#
# 用法：
#   bash run_experiments.sh          # 顺序运行 baseline + ggeur
#   bash run_experiments.sh baseline # 只运行 baseline
#   bash run_experiments.sh ggeur    # 只运行 ggeur
# =============================================================================

set -e

PYTHON=/root/.local/share/mamba/envs/fs/bin/python3
WORK_DIR=/root/lzw/FederatedScope
LOG_DIR=${WORK_DIR}/exp/logs

# 输出目录
BASELINE_OUT=${WORK_DIR}/exp/baseline_allclients
GGEUR_OUT=${WORK_DIR}/exp/ggeur_allclients

# 配置文件
BASELINE_CFG=${WORK_DIR}/scripts/fedmia_offline_features.yaml
GGEUR_CFG=${WORK_DIR}/scripts/ggeur_fedmia_offline_features.yaml

# =============================================================================

cd ${WORK_DIR}
mkdir -p ${LOG_DIR}

run_baseline() {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Baseline 实验"
    echo "  配置: ${BASELINE_CFG}"
    echo "  输出: ${BASELINE_OUT}"
    echo "============================================================"

    mkdir -p ${BASELINE_OUT}

    nohup ${PYTHON} federatedscope/main.py \
        --cfg ${BASELINE_CFG} \
        outdir ${BASELINE_OUT} \
        expname fedmia_baseline \
        > ${LOG_DIR}/baseline.log 2>&1 &

    BASELINE_PID=$!
    echo "[Baseline] PID: ${BASELINE_PID}"
    echo "[Baseline] 日志: ${LOG_DIR}/baseline.log"
    echo ${BASELINE_PID} > ${LOG_DIR}/baseline.pid
}

run_ggeur() {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 GGEUR 实验"
    echo "  配置: ${GGEUR_CFG}"
    echo "  输出: ${GGEUR_OUT}"
    echo "============================================================"

    mkdir -p ${GGEUR_OUT}

    nohup ${PYTHON} federatedscope/main.py \
        --cfg ${GGEUR_CFG} \
        outdir ${GGEUR_OUT} \
        expname fedmia_ggeur \
        > ${LOG_DIR}/ggeur.log 2>&1 &

    GGEUR_PID=$!
    echo "[GGEUR]    PID: ${GGEUR_PID}"
    echo "[GGEUR]    日志: ${LOG_DIR}/ggeur.log"
    echo ${GGEUR_PID} > ${LOG_DIR}/ggeur.pid
}

wait_and_check() {
    echo ""
    echo "============================================================"
    echo "实验监控命令："
    echo "  tail -f ${LOG_DIR}/baseline.log"
    echo "  tail -f ${LOG_DIR}/ggeur.log"
    echo ""
    echo "查看进程："
    echo "  ps aux | grep main.py"
    echo ""
    echo "终止实验："
    echo "  kill \$(cat ${LOG_DIR}/baseline.pid)"
    echo "  kill \$(cat ${LOG_DIR}/ggeur.pid)"
    echo "============================================================"
}

# =============================================================================
# 主流程
# =============================================================================

TARGET=${1:-all}

case ${TARGET} in
    baseline)
        run_baseline
        ;;
    ggeur)
        run_ggeur
        ;;
    all|*)
        run_baseline
        sleep 2
        run_ggeur
        ;;
esac

wait_and_check
