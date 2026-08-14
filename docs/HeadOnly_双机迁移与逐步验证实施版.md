# HeadOnly 双机迁移与逐步验证实施版

本文是一份迁移后可以立即实施的执行顺序。目标是在两台 IPv6 公网互通机器上完成：

1. 代码迁移和 Python 环境检查
2. 双机 QPS 可行性 smoke test
3. QPS 阶梯压测并找当前性能下的最大成功 QPS
4. FederatedScope/gRPC HeadOnly smoke test

当前地址：

```text
服务器 IPv6: 2001:da8:215:6a01:be24:11ff:fe58:570d
本地电脑 IPv6: 2001:da8:215:3c0a:f51:4d71:80:8075
```

QPS 脚本是 TCP/asyncio，IPv6 不加方括号。FederatedScope/gRPC 配置里 IPv6 要加方括号。

## 0. 推荐验证顺序

先做 QPS，再做完整训练：

```text
QPS 1x10 smoke
QPS 1x50
QPS 2x100
QPS 4x250
QPS 4x500
QPS 4x1000
HeadOnly/gRPC 2 clients 1 round
```

QPS 不依赖数据集、CLIP、torch，最适合先验证双机链路和参数下发能力。

## 1. 在开发机打包项目

开发机 PowerShell：

```powershell
cd D:\Projects\FederatedScope
powershell -ExecutionPolicy Bypass -File scripts\package_headonly_two_machine_bundle.ps1
```

输出文件默认是：

```text
%TEMP%\FederatedScope_headonly_two_machine.zip
```

查看路径：

```powershell
Write-Host "$env:TEMP\FederatedScope_headonly_two_machine.zip"
```

## 2. 上传到服务器

PowerShell：

```powershell
scp "$env:TEMP\FederatedScope_headonly_two_machine.zip" <SERVER_SSH>:/root/autodl-tmp/
```

服务器：

```bash
mkdir -p /root/autodl-tmp/FederatedScope
cd /root/autodl-tmp/FederatedScope
unzip -o /root/autodl-tmp/FederatedScope_headonly_two_machine.zip
```

检查：

```bash
python3 -m py_compile scripts/benchmark_headonly_mlp_download_window.py
python3 -m py_compile scripts/summarize_headonly_qps_runs.py
ls scripts/distributed_scripts/ggeur_headonly_download_qps
```

## 3. 放到本地验证机器

如果本地验证机器就是当前 Windows 机器，可以直接使用 `D:\Projects\FederatedScope`。

推荐用 WSL/Linux 跑 Bash 脚本。把 zip 解压到 WSL：

```powershell
wsl mkdir -p /root/autodl-tmp/FederatedScope
wsl sh -lc 'rm -rf /root/autodl-tmp/FederatedScope/*'
```

把下面路径按你的真实 `%TEMP%` 位置调整：

```powershell
wsl unzip -o /mnt/c/Users/$env:USERNAME/AppData/Local/Temp/FederatedScope_headonly_two_machine.zip -d /root/autodl-tmp/FederatedScope
```

WSL 检查：

```bash
cd /root/autodl-tmp/FederatedScope
python3 -m py_compile scripts/benchmark_headonly_mlp_download_window.py
python3 -m py_compile scripts/summarize_headonly_qps_runs.py
```

如果没有 WSL，也可以用 Windows Python 跑 QPS client，见本文第 8 节。

## 4. 放行服务器 QPS 端口

QPS 阶梯脚本最多使用：

```text
39301
39302
39303
39304
```

服务器如果有防火墙：

```bash
sudo ufw allow 39301:39304/tcp
sudo ufw status
```

云平台安全组也要放行 TCP `39301-39304`。

快速测试：

服务器：

```bash
python3 -m http.server 39301 --bind ::
```

本地 PowerShell：

```powershell
Test-NetConnection 2001:da8:215:6a01:be24:11ff:fe58:570d -Port 39301
```

成功后服务器按 `Ctrl+C` 退出。

## 5. QPS smoke test

先只跑 1 个 subserver、10 个 logical clients。

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_smoke \
QPS_PLAN='smoke_1x10:1:10:120:120' \
LISTEN_HOST='::' \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_server.sh
```

本地 WSL/Linux：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_smoke \
QPS_PLAN='smoke_1x10:1:10:120:120' \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_clients.sh
```

成功标准：

```text
success_ratio = 1.0
completed_clients = expected_clients = 10
```

服务器查看：

```bash
cat exp/headonly_download_qps/qps_ipv6_smoke_summary.tsv
```

## 6. QPS 阶梯压测

服务器先启动：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_profile_$(date +%Y%m%d_%H%M%S) \
LISTEN_HOST='::' \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_server.sh
```

记下服务器打印的 `run_prefix`。例如：

```text
qps_ipv6_profile_20260611_153000
```

本地 WSL/Linux 用同一个 `RUN_PREFIX`：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_profile_20260611_153000 \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_clients.sh
```

默认阶梯：

```text
smoke_1x10
small_1x50
medium_2x100
large_4x250
max_4x500
max_4x1000
```

服务器汇总：

```bash
cat exp/headonly_download_qps/qps_ipv6_profile_20260611_153000_summary.tsv
```

汇总文件末尾的 `# best_successful_run` 是当前阶梯里成功且 QPS 最高的一档。

## 7. 更激进的 QPS 最大值搜索

如果 `4x1000` 成功，可以继续加大。

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_max_$(date +%Y%m%d_%H%M%S) \
QPS_PLAN='max_4x1500:4:1500:900:900,max_4x2000:4:2000:1200:1200,max_8x1000:8:1000:1200:1200' \
LISTEN_HOST='::' \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_server.sh
```

如果跑到 `8x1000`，服务器需要放行端口 `39301-39308`。

本地：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_PREFIX=qps_ipv6_max_20260611_153000 \
QPS_PLAN='max_4x1500:4:1500:900:900,max_4x2000:4:2000:1200:1200,max_8x1000:8:1000:1200:1200' \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
BASE_PORT=39301 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_ipv6_qps_plan_clients.sh
```

最大 QPS 的判定不是 client 数越大越好，而是在 `success_ratio=1.0` 的成功结果中，`global_download_qps_go_to_last` 最大的那一档。

## 8. 没有 WSL 时的 Windows QPS client

服务器仍按第 5 节启动。

本地 PowerShell：

```powershell
cd D:\Projects\FederatedScope

python scripts\benchmark_headonly_mlp_download_window.py client `
  --connect-host 2001:da8:215:6a01:be24:11ff:fe58:570d `
  --base-port 39301 `
  --subservers 1 `
  --clients-per-subserver 10 `
  --connect-timeout 120
```

阶梯压测建议使用 WSL，因为 Bash 脚本会自动跑多档并汇总。

## 9. gRPC HeadOnly smoke test 准备

QPS 成功后，再验证完整 FederatedScope/gRPC 双机训练。

注意：

```text
gRPC IPv6 要加方括号
QPS IPv6 不加方括号
```

服务器生成配置：

```bash
cd /root/autodl-tmp/FederatedScope

python3 scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id two_machine_grpc_smoke \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --head-only \
  --headonly-eval-mode client \
  --server-host '[2001:da8:215:6a01:be24:11ff:fe58:570d]' \
  --server-bind-host '[::]' \
  --client-host '[2001:da8:215:3c0a:f51:4d71:80:8075]' \
  --server-port 51251 \
  --client-port-base 52250 \
  --clients 2 \
  --sample-clients 2 \
  --rounds 1 \
  --num-generated-per-sample 1 \
  --num-generated-per-prototype 1 \
  --target-size-per-class 1 \
  --feature-cache-version two_machine_grpc_smoke_v1
```

如果本地 GPU 显存小，先不要加大 batch 或生成样本数。必要时生成后手动把 client YAML 的：

```yaml
dataloader:
  batch_size: 8

ggeur:
  extract_batch_size: 8
```

## 10. gRPC smoke test 启动

先确保两台机器都有数据集和模型权重；QPS 不需要，gRPC HeadOnly 训练需要。

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly
```

本地：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=python3 \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly
```

成功日志关键字：

```bash
grep -E "Client #[0-9]+ has joined|Received statistics quorum|Broadcasting global covariances|Broadcasting round|Training finished" \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_grpc_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

## 11. 最终建议

正式记录结果时分开写：

```text
QPS benchmark:
  TCP/asyncio, no training, no dataset, no gRPC

HeadOnly distributed smoke:
  FederatedScope/gRPC, real join/broadcast/train/eval lifecycle
```

不要把 SSH tunnel 结果、loopback 结果、IPv6 双机结果混在同一张 QPS 表里。
