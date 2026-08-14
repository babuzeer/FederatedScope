# HeadOnly QPS 双机从零部署执行版

本文用于在两台当前没有项目代码的机器上，逐步完成 HeadOnly 参数下发 QPS 双机验证。

当前已知网络条件：

```text
服务器 IPv6:
2001:da8:215:6a01:be24:11ff:fe58:570d

本地电脑稳定 IPv6:
2001:da8:215:3c0a:f51:4d71:80:8075
```

QPS benchmark 是纯 TCP/asyncio，不是 FederatedScope gRPC。因此本文件中的 QPS 命令使用裸 IPv6 地址，不加方括号。

```text
正确:
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d
LISTEN_HOST=::

不要写:
CONNECT_HOST=[2001:da8:215:6a01:be24:11ff:fe58:570d]
```

## 0. 目标拓扑

推荐先用服务器跑 QPS server，本地电脑跑 QPS clients：

```text
服务器:
  subserver listeners: [::]:39301, [::]:39302, ...

本地电脑:
  logical clients connect to 2001:da8:215:6a01:be24:11ff:fe58:570d:39301...
```

这条链路测的是：

```text
服务器 -> 本地电脑
```

也就是参数下发方向，适合 QPS 验证。

## 1. 准备仓库代码

两台机器必须有同一份代码。由于你当前本地工作区有未提交改动，最稳妥是先在当前开发机把工作区打包，再复制到服务器和本地验证机器。

### 1.1 在当前开发机打包项目

PowerShell，在当前项目目录执行：

```powershell
cd D:\Projects\FederatedScope
git status --short
```

确认这是你要部署的版本后，打包。排除 `.git`、实验输出和数据集：

```powershell
$src = "D:\Projects\FederatedScope"
$dst = "$env:TEMP\FederatedScope_headonly_qps.zip"
if (Test-Path $dst) { Remove-Item $dst }
Compress-Archive -Path `
  "$src\benchmark", `
  "$src\doc", `
  "$src\docs", `
  "$src\environment", `
  "$src\federatedscope", `
  "$src\scripts", `
  "$src\tests", `
  "$src\README.md", `
  "$src\setup.py", `
  "$src\run.py", `
  "$src\LICENSE", `
  "$src\meta.yaml" `
  -DestinationPath $dst
Write-Host $dst
```

如果压缩命令因为路径缺失失败，先确认这些目录是否存在：

```powershell
Get-ChildItem D:\Projects\FederatedScope
```

### 1.2 上传代码到服务器

PowerShell：

```powershell
scp $env:TEMP\FederatedScope_headonly_qps.zip <SERVER_SSH>:/root/autodl-tmp/
```

服务器解压：

```bash
mkdir -p /root/autodl-tmp/FederatedScope
cd /root/autodl-tmp/FederatedScope
unzip -o /root/autodl-tmp/FederatedScope_headonly_qps.zip
```

检查 QPS 脚本存在：

```bash
ls scripts/distributed_scripts/ggeur_headonly_download_qps
ls scripts/benchmark_headonly_mlp_download_window.py
```

### 1.3 把代码放到本地验证机器

如果本地验证机器就是当前 Windows 电脑，可以直接使用 `D:\Projects\FederatedScope`。

如果本地要用 WSL/Linux 跑 clients，推荐同步到 WSL：

```powershell
wsl mkdir -p /root/autodl-tmp/FederatedScope
wsl rm -rf /root/autodl-tmp/FederatedScope/*
wsl unzip -o /mnt/c/Users/$env:USERNAME/AppData/Local/Temp/FederatedScope_headonly_qps.zip -d /root/autodl-tmp/FederatedScope
```

如果 `$env:TEMP` 不在上述路径，先打印真实路径：

```powershell
Write-Host $env:TEMP
```

然后把 zip 路径改成对应的 `/mnt/c/...` 路径。

## 2. 准备 Python 环境

QPS benchmark 只需要 Python 标准库，不依赖 torch，不需要数据集，不需要 CLIP 权重。

服务器：

```bash
cd /root/autodl-tmp/FederatedScope
python3 --version
python3 -m py_compile scripts/benchmark_headonly_mlp_download_window.py
```

本地 WSL/Linux：

```bash
cd /root/autodl-tmp/FederatedScope
python3 --version
python3 -m py_compile scripts/benchmark_headonly_mlp_download_window.py
```

如果本地不用 WSL，也可以用 Windows Python 直接运行 `scripts/benchmark_headonly_mlp_download_window.py`，但现有 `run_download_qps_clients.sh` 是 Bash 脚本，推荐用 WSL。

## 3. 再确认 QPS 端口可达

QPS server 默认从 `39301` 开始监听。先测一个端口。

服务器：

```bash
python3 -m http.server 39301 --bind ::
```

本地 PowerShell：

```powershell
Test-NetConnection 2001:da8:215:6a01:be24:11ff:fe58:570d -Port 39301
```

成功标准：

```text
TcpTestSucceeded : True
```

成功后，在服务器按 `Ctrl+C` 退出临时 http server。

如果失败，需要在服务器防火墙或安全组放行 QPS 端口，例如 `39301-39308/tcp`。

## 4. 运行最小 QPS smoke test

### 4.1 服务器启动 QPS server

服务器窗口 A：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_smoke_1x10_server \
LISTEN_HOST='::' \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=10 \
BASE_PORT=39301 \
READY_TIMEOUT=120 \
ACK_TIMEOUT=120 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh
```

这个命令会等待 clients 连接，不要关闭。

### 4.2 本地启动 QPS clients

本地 WSL/Linux：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_smoke_1x10_clients \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=10 \
BASE_PORT=39301 \
CONNECT_TIMEOUT=120 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh
```

成功后，server 会输出 summary 路径，例如：

```text
exp/headonly_download_qps/qps_ipv6_smoke_1x10_server
```

## 5. 查看结果

服务器：

```bash
cd /root/autodl-tmp/FederatedScope
cat exp/headonly_download_qps/qps_ipv6_smoke_1x10_server/download_qps_summary.json
```

快速提取关键指标：

```bash
python3 - <<'PY'
import json
path = 'exp/headonly_download_qps/qps_ipv6_smoke_1x10_server/download_qps_summary.json'
data = json.load(open(path, 'r', encoding='utf-8'))
g = data['global']
print('expected_clients=', g['expected_clients'])
print('completed_clients=', g['completed_clients'])
print('success_ratio=', g['success_ratio'])
print('global_download_qps_go_to_last=', g['global_download_qps_go_to_last'])
print('global_download_window_go_to_last_sec=', g['global_download_window_go_to_last_sec'])
print('payload_bytes_per_download=', g['payload_bytes_per_download'])
print('total_payload_bytes=', g['total_payload_bytes'])
PY
```

通过标准：

```text
completed_clients = expected_clients
success_ratio = 1.0
global_download_qps_go_to_last > 0
```

## 6. 逐步加压

每次只改一档，确认成功后再继续。

### 6.1 1 个 subserver，50 clients

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_1x50_server \
LISTEN_HOST='::' \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=50 \
BASE_PORT=39301 \
READY_TIMEOUT=180 \
ACK_TIMEOUT=180 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh
```

本地：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_1x50_clients \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
SUBSERVERS=1 \
CLIENTS_PER_SUBSERVER=50 \
BASE_PORT=39301 \
CONNECT_TIMEOUT=180 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh
```

### 6.2 2 个 subserver，各 100 clients

需要服务器放行端口：

```text
39301
39302
```

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_2x100_server \
LISTEN_HOST='::' \
SUBSERVERS=2 \
CLIENTS_PER_SUBSERVER=100 \
BASE_PORT=39301 \
READY_TIMEOUT=240 \
ACK_TIMEOUT=240 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh
```

本地：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_2x100_clients \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
SUBSERVERS=2 \
CLIENTS_PER_SUBSERVER=100 \
BASE_PORT=39301 \
CONNECT_TIMEOUT=240 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh
```

### 6.3 4 个 subserver，各 250 clients

需要服务器放行端口：

```text
39301
39302
39303
39304
```

服务器：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_4x250_server \
LISTEN_HOST='::' \
SUBSERVERS=4 \
CLIENTS_PER_SUBSERVER=250 \
BASE_PORT=39301 \
READY_TIMEOUT=300 \
ACK_TIMEOUT=300 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_server.sh
```

本地：

```bash
cd /root/autodl-tmp/FederatedScope

RUN_ID=qps_ipv6_4x250_clients \
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d \
SUBSERVERS=4 \
CLIENTS_PER_SUBSERVER=250 \
BASE_PORT=39301 \
CONNECT_TIMEOUT=300 \
PYTHON_BIN=python3 \
bash scripts/distributed_scripts/ggeur_headonly_download_qps/run_download_qps_clients.sh
```

## 7. 端口和进程检查

服务器查看监听：

```bash
ss -ltnp | grep -E '39301|39302|39303|39304' || true
```

服务器查看结果目录：

```bash
find exp/headonly_download_qps -maxdepth 2 -type f | sort
```

本地查看 client 日志：

```bash
cd /root/autodl-tmp/FederatedScope
cat exp/headonly_download_qps/qps_ipv6_smoke_1x10_clients/clients.log
```

## 8. 常见问题

### 8.1 client 连接超时

检查服务器监听和防火墙：

```bash
ss -ltnp | grep 39301 || true
```

本地测端口：

```powershell
Test-NetConnection 2001:da8:215:6a01:be24:11ff:fe58:570d -Port 39301
```

### 8.2 server 一直等待 clients

说明 client 没有连进来，或 `SUBSERVERS` / `CLIENTS_PER_SUBSERVER` 两边不一致。确保 server 和 client 两边参数完全一致：

```text
SUBSERVERS
CLIENTS_PER_SUBSERVER
BASE_PORT
```

### 8.3 IPv6 写成方括号导致失败

QPS 脚本不要加方括号。

错误：

```bash
CONNECT_HOST='[2001:da8:215:6a01:be24:11ff:fe58:570d]'
```

正确：

```bash
CONNECT_HOST=2001:da8:215:6a01:be24:11ff:fe58:570d
```

### 8.4 本地没有 WSL

推荐安装 WSL 后执行 Bash 脚本。临时替代方案是直接用 Windows Python 跑 client：

```powershell
cd D:\Projects\FederatedScope
python scripts\benchmark_headonly_mlp_download_window.py client `
  --connect-host 2001:da8:215:6a01:be24:11ff:fe58:570d `
  --base-port 39301 `
  --subservers 1 `
  --clients-per-subserver 10 `
  --connect-timeout 120
```

server 端仍按第 4.1 节运行。

## 9. 记录实验结果

每次实验记录：

```text
run_id
服务器 IP
本地 client IP
SUBSERVERS
CLIENTS_PER_SUBSERVER
completed_clients
success_ratio
global_download_qps_go_to_last
global_download_window_go_to_last_sec
payload_bytes_per_download
total_payload_bytes
```

建议先保留这些 run：

```text
qps_ipv6_smoke_1x10
qps_ipv6_1x50
qps_ipv6_2x100
qps_ipv6_4x250
```
