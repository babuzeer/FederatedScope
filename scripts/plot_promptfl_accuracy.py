"""
绘制 GGEUR PromptFL 准确率曲线

用法:
    python scripts/plot_promptfl_accuracy.py                    # 绘制两个实验
    python scripts/plot_promptfl_accuracy.py --exp officehome   # 只绘制 OfficeHome
    python scripts/plot_promptfl_accuracy.py --exp pacs         # 只绘制 PACS
"""
import re
import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ========== 配置（与 yaml 保持一致）==========
SAVE_DIR = 'exp/ggeur_promptfl_clip'

EXPERIMENTS = {
    'officehome': {
        'outdir': 'exp/ggeur_promptfl_clip',
        'expname': 'ggeur_promptfl_officehome',
        'title': 'GGEUR PromptFL - OfficeHome (60 clients, α=0.1)',
        'domains': ['Art', 'Clipart', 'Product', 'Real_World'],
    },
    'pacs': {
        'outdir': 'exp/ggeur_promptfl_clip',
        'expname': 'ggeur_promptfl_pacs',
        'title': 'GGEUR PromptFL - PACS (4 clients, α=0.1)',
        'domains': ['photo', 'art_painting', 'cartoon', 'sketch'],
    },
}

# 日志行模式
PATTERN_MLP    = re.compile(r'Round (\d+) MLP Test Accuracy - (.+)')
PATTERN_PROMPT = re.compile(r'Round (\d+) Prompt Test Accuracy - (.+)')
PATTERN_KV     = re.compile(r'(\w+): ([\d.]+)')


def find_latest_log(outdir: str, expname: str) -> str:
    """找到最新一次运行的 exp_print.log"""
    base = os.path.join(outdir, expname)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"实验目录不存在: {base}")

    # 优先找 sub_exp_* 子目录（按名称排序，最新在最后）
    sub_dirs = sorted(glob.glob(os.path.join(base, 'sub_exp_*')))
    if sub_dirs:
        log_path = os.path.join(sub_dirs[-1], 'exp_print.log')
        if os.path.isfile(log_path):
            print(f"  使用子目录: {sub_dirs[-1]}")
            return log_path

    # 否则用根目录下的 log
    log_path = os.path.join(base, 'exp_print.log')
    if os.path.isfile(log_path):
        print(f"  使用目录: {base}")
        return log_path

    raise FileNotFoundError(f"找不到 exp_print.log，已检查: {base}")


def parse_log(log_path: str):
    """
    解析日志，返回:
        mlp_data:    {round: {domain: acc, 'average': acc}}
        prompt_data: {round: {domain: acc, 'average': acc}}
    """
    mlp_data, prompt_data = {}, {}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            for pattern, store in [(PATTERN_MLP, mlp_data),
                                   (PATTERN_PROMPT, prompt_data)]:
                m = pattern.search(line)
                if m:
                    rnd = int(m.group(1))
                    kv = {k: float(v) for k, v in PATTERN_KV.findall(m.group(2))}
                    store[rnd] = kv

    return mlp_data, prompt_data


def plot_experiment(exp_key: str, cfg: dict, save_dir: str):
    outdir  = cfg['outdir']
    expname = cfg['expname']
    title   = cfg['title']

    print(f"\n[{exp_key}] 查找日志...")
    log_path = find_latest_log(outdir, expname)
    print(f"  日志: {log_path}")

    mlp_data, prompt_data = parse_log(log_path)

    has_mlp    = len(mlp_data) > 0
    has_prompt = len(prompt_data) > 0

    if not has_mlp and not has_prompt:
        print(f"  [警告] 未找到任何准确率数据，跳过")
        return

    print(f"  MLP rounds: {len(mlp_data)},  Prompt rounds: {len(prompt_data)}")

    # ---- 确定子图数量 ----
    n_plots = (1 if has_mlp else 0) + (1 if has_prompt else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plot_idx = 0
    colors = plt.cm.tab10.colors

    for label, data, has in [('MLP Classifier', mlp_data, has_mlp),
                              ('PromptFL (Soft Prompt)', prompt_data, has_prompt)]:
        if not has:
            continue
        ax = axes[plot_idx]
        plot_idx += 1

        rounds = sorted(data.keys())
        all_keys = list(data[rounds[0]].keys())
        domain_keys = [k for k in all_keys if k != 'average']

        # 各 domain 曲线（细线）
        for i, domain in enumerate(domain_keys):
            vals = [data[r].get(domain, float('nan')) * 100 for r in rounds]
            ax.plot(rounds, vals, color=colors[i % len(colors)],
                    linewidth=1.2, alpha=0.7, linestyle='--',
                    label=domain)

        # average 曲线（粗线）
        if 'average' in data[rounds[0]]:
            avg_vals = [data[r]['average'] * 100 for r in rounds]
            best_acc = max(avg_vals)
            best_rnd = rounds[np.argmax(avg_vals)]
            ax.plot(rounds, avg_vals, color='black',
                    linewidth=2.2, label=f'Average (best={best_acc:.2f}%)')
            ax.axhline(best_acc, color='black', linewidth=0.8,
                       linestyle=':', alpha=0.5)
            ax.annotate(f'{best_acc:.2f}%',
                        xy=(best_rnd, best_acc),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black')
            print(f"  [{label}] best avg: {best_acc:.2f}% @ round {best_rnd}")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        y_min = max(0, min(
            v for r in rounds for v in [data[r].get(k, 100) * 100
                                        for k in all_keys]) - 5)
        ax.set_ylim(bottom=y_min)

    plt.tight_layout()
    out_png = os.path.join(save_dir, f'promptfl_{exp_key}.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"  已保存: {out_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=['officehome', 'pacs', 'all'],
                        default='all', help='要绘制的实验')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    keys = list(EXPERIMENTS.keys()) if args.exp == 'all' else [args.exp]
    for key in keys:
        try:
            plot_experiment(key, EXPERIMENTS[key], SAVE_DIR)
        except FileNotFoundError as e:
            print(f"  [跳过] {e}")

    print("\n完成。")


if __name__ == '__main__':
    main()
