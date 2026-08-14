# HeadOnly 双机通信与多机验证操作手册

本文用于把一台本地电脑和一台可通信服务器接入 FederatedScope/GGEUR HeadOnly 的真实分布式验证流程。

核心原则：FederatedScope distributed 需要双向 TCP 可达。client 会主动连接 server 发送 `join_in`，同时 client 会把自己的 `client_host:client_port` 上报给 server，server 后续会回连 client 下发 `assign_client_id`、模型参数、协方差、finish 等消息。

因此，开始训练前必须同时验证：

```text
本地电脑 -> 服务器:server_port
服务器 -> 本地电脑:client_port
```

如果本地电脑在家庭/办公室 NAT 后面，服务器大概率不能直接访问本地电脑内网 IP。此时优先使用 Tailscale/ZeroTier/VPN，让两台机器获得同一虚拟内网 IP，再把这些虚拟 IP 写入配置。

## 0. 填写你的机器信息

后续命令统一使用这些变量。把尖括号内容替换成你的实际值。

```text
SERVER_IP=<服务器可被本地电脑访问的IP>
CLIENT_IP=<本地电脑可被服务器访问的IP>
SERVER_SSH=<服务器SSH别名或user@host>
SERVER_PORT=51251
CLIENT_PORT_BASE=52250
CLIENT_1_PORT=52251
REPO_DIR=/root/autodl-tmp/FederatedScope
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python
```

`SERVER_IP` 和 `CLIENT_IP` 必须是通信对端能访问到的地址。不要使用 `127.0.0.1`，除非 server 和 client 在同一台机器上。

## 1. 获取两台机器 IP

### 1.1 服务器上查看 IP

```bash
hostname -I
curl -s ifconfig.me && echo
```

如果使用 Tailscale：

```bash
tailscale ip -4
```

### 1.2 本地 Windows 查看 IP

PowerShell：

```powershell
ipconfig
```

如果使用 Tailscale：

```powershell
tailscale ip -4
```

记录最终要用的 `SERVER_IP` 和 `CLIENT_IP`。

## 2. 验证本地电脑能访问服务器

### 2.1 服务器临时监听 server 端口

服务器执行：

```bash
python3 -m http.server 51251 --bind 0.0.0.0
```

如果服务器开启了防火墙，需要放行端口。Ubuntu 常见命令：

```bash
sudo ufw allow 51251/tcp
sudo ufw status
```

云服务器还需要在云厂商安全组中放行 TCP `51251`。

### 2.2 本地 Windows 测试连接服务器

PowerShell：

```powershell
Test-NetConnection <SERVER_IP> -Port 51251
```

通过标准：

```text
TcpTestSucceeded : True
```

也可以用 Python 测：

```powershell
python -c "import socket; socket.create_connection(('<SERVER_IP>', 51251), 5); print('ok')"
```

测试完成后，在服务器监听窗口按 `Ctrl+C` 退出。

## 3. 验证服务器能访问本地电脑

这是最容易失败的一步，也是 FederatedScope 双机验证的关键。

### 3.1 本地 Windows 临时监听 client 端口

PowerShell：

```powershell
$listener = [System.Net.Sockets.TcpListener]::new([Net.IPAddress]::Any, 52251)
$listener.Start()
"listening on 52251"
$client = $listener.AcceptTcpClient()
"connected"
$client.Close()
$listener.Stop()
```

如果 Windows 防火墙拦截，先放行端口：

```powershell
New-NetFirewallRule -DisplayName "FederatedScope client 52251" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 52251
```

### 3.2 服务器测试连接本地电脑

服务器执行：

```bash
python3 -c "import socket; socket.create_connection(('<CLIENT_IP>', 52251), 5); print('ok')"
```

通过标准：

```text
ok
```

并且本地 PowerShell 监听窗口打印：

```text
connected
```

如果失败，按顺序检查：

1. `CLIENT_IP` 是否是服务器可访问的地址，而不是本地局域网私有地址。
2. Windows 防火墙是否放行 `52251/tcp`。
3. 本地路由器/NAT 是否阻止入站连接。
4. 是否需要使用 Tailscale/ZeroTier/VPN。

## 4. 推荐网络方案

### 4.1 服务器有公网 IP，本地电脑无公网 IP

推荐使用 Tailscale 或 ZeroTier。

配置完成后：

```text
SERVER_IP = 服务器的 Tailscale/ZeroTier IP
CLIENT_IP = 本地电脑的 Tailscale/ZeroTier IP
```

然后重新执行第 2、3 节双向 TCP 测试。

### 4.2 两台机器在同一局域网

可以直接使用局域网 IP。

```text
SERVER_IP = 服务器局域网 IP
CLIENT_IP = 本地电脑局域网 IP
```

仍然必须通过第 2、3 节测试。

### 4.3 只有本地电脑能访问服务器，服务器不能访问本地电脑

不能直接跑 FederatedScope 双机 distributed。需要先解决双向连通问题，否则 server 无法给 client 下发消息。

## 5. 准备代码和环境

两台机器上需要有同一份 FederatedScope 代码和可运行 Python 环境。

服务器示例：

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> FederatedScope
cd FederatedScope
git status --short
```

如果本地电脑已经有代码，可以同步到服务器：

```powershell
scp -r D:\Projects\FederatedScope <SERVER_SSH>:/root/autodl-tmp/
```

服务器验证 Python 环境：

```bash
cd /root/autodl-tmp/FederatedScope
/root/miniconda3/envs/fs/bin/python -c "import torch; print(torch.__version__)"
/root/miniconda3/envs/fs/bin/python -m py_compile federatedscope/main.py
```

本地电脑如果要运行 client，也需要能执行 `federatedscope/main.py`。建议在 WSL/Linux 环境运行 client；如果用 Windows 原生 Python，需要手动启动，不使用 Bash runner。

## 6. 准备数据和模型

HeadOnly OfficeHome + ViT 默认依赖：

```text
OfficeHome 数据集:
  /root/autodl-tmp/datasets/OfficeHomeDataset_10072016

OpenCLIP ViT-B/16 权重:
  /root/autodl-tmp/models/open_clip_vitb16.bin
```

服务器和本地 client 机器都需要能访问自己的本地数据和模型路径。如果路径不同，生成配置时用 `--data-root` 指定数据路径，或生成后手动改 YAML。

快速验证文件存在：

服务器：

```bash
ls /root/autodl-tmp/datasets/OfficeHomeDataset_10072016
ls /root/autodl-tmp/models/open_clip_vitb16.bin
```

本地 Linux/WSL client：

```bash
ls /root/autodl-tmp/datasets/OfficeHomeDataset_10072016
ls /root/autodl-tmp/models/open_clip_vitb16.bin
```

## 7. 生成一个最小双机 HeadOnly case

在控制端机器上执行。建议先在服务器仓库里生成，然后把生成目录同步到本地 client 机器。

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

/root/miniconda3/envs/fs/bin/python \
  scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id two_machine_smoke \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --head-only \
  --server-host <SERVER_IP> \
  --server-bind-host 0.0.0.0 \
  --client-host <CLIENT_IP> \
  --server-port 51251 \
  --client-port-base 52250 \
  --clients 2 \
  --sample-clients 2 \
  --rounds 1 \
  --num-generated-per-sample 1 \
  --num-generated-per-prototype 1 \
  --target-size-per-class 1 \
  --feature-cache-version two_machine_smoke_v1
```

生成目录：

```text
scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly/
```

检查配置中的 IP：

```bash
grep -n "server_host\|server_port\|client_host\|client_port\|role" \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly/configs/*.yaml
```

期望：

```text
server.yaml:
  role: server
  server_host: 0.0.0.0
  server_port: 51251

client_1.yaml / client_2.yaml:
  role: client
  server_host: <SERVER_IP>
  server_port: 51251
  client_host: <CLIENT_IP>
  client_port: 52251 / 52252
```

## 8. 同步 case 到本地 client 机器

如果 server 和 client 使用同一路径 `/root/autodl-tmp/FederatedScope`，只需要同步整个仓库或至少同步生成目录。

从服务器同步到本地 Linux/WSL client 的一种方式：

```bash
scp -r <SERVER_SSH>:/root/autodl-tmp/FederatedScope/scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke \
  /root/autodl-tmp/FederatedScope/scripts/distributed_scripts/ggeur_multimachine/runs/
```

如果本地是 Windows 原生路径，需要确保配置里的 `outdir`、数据路径、模型路径对本地 Python 可用。更推荐用 WSL 保持 Linux 路径一致。

## 9. 手动启动双机验证

这种方式最直观，适合第一次排错。

### 9.1 服务器启动 server

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly
```

查看 server 日志：

```bash
tail -f scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

需要看到类似：

```text
Listen to 0.0.0.0:51251
```

### 9.2 本地 client 机器启动 clients

本地 Linux/WSL 执行：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly
```

查看 client 日志：

```bash
tail -f scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly/logs/client_1.log
```

## 10. 判定是否成功

服务器日志应出现：

```text
Server: Client #1 has joined in
Server: Client #2 has joined in
Server: Received statistics quorum
Server: Broadcasting global covariances
Server: Broadcasting round
Server: Training finished
```

client 日志应出现：

```text
Client: Listen to <CLIENT_IP>:52251
Client 1: ...
augmentation ready
```

汇总 case：

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

/root/miniconda3/envs/fs/bin/python \
  scripts/distributed_scripts/ggeur_multimachine/summarize_case.py \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly
```

如果需要停止残留进程：

```bash
cd /root/autodl-tmp/FederatedScope

bash scripts/distributed_scripts/ggeur_multimachine/stop_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly
```

server 和 client 机器都执行一次 stop 更稳妥。

## 11. 使用远程 runner 自动启动

前提：

1. 控制端能 SSH 到 server 机器。
2. 控制端能 SSH 到 client 机器。
3. 两台远端机器上的仓库路径一致，例如都是 `/root/autodl-tmp/FederatedScope`。
4. 生成后的 case 目录在两台机器上都存在。

如果本地电脑就是 client 机器，但没有 SSH server，先用第 9 节手动启动，不用本节。

控制端执行：

```bash
cd /root/autodl-tmp/FederatedScope

REPO_DIR=/root/autodl-tmp/FederatedScope \
PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
SERVER_READY_TIMEOUT=300 \
CASE_TIMEOUT=3600 \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/run_remote_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_smoke/officehome_vit_fedavg_headonly \
  <SERVER_SSH> \
  <CLIENT_SSH>
```

## 12. 常见问题

### 12.1 client 能 join，但收不到 server 下发

通常是 server 无法访问 `client_host:client_port`。

检查：

```bash
python3 -c "import socket; socket.create_connection(('<CLIENT_IP>', 52251), 5); print('ok')"
```

如果不通，回到第 3 节。

### 12.2 server 一直等待 client join

检查 client 是否能访问 `server_host:server_port`：

```bash
python3 -c "import socket; socket.create_connection(('<SERVER_IP>', 51251), 5); print('ok')"
```

同时看 client 日志中是否有连接错误。

### 12.3 日志里出现 join timeout

说明 `federate.client_num` 个 client 没有全部完成 join。先用 `--clients 1` 或 `--clients 2` 做 smoke test，不要一开始跑完整矩阵。

### 12.4 本地电脑是 Windows，Bash runner 不能运行

推荐用 WSL。Windows 原生 PowerShell 可以做通信测试，但当前多机 runner 是 Bash 脚本，直接在 Windows PowerShell 下不可用。

### 12.5 配置里误写了 127.0.0.1

双机验证不能把 `server_host` 或 `client_host` 写成 `127.0.0.1`。这只会指向进程所在机器自己。

## 13. 通过 smoke test 后再扩大规模

先把 smoke test 跑通：

```text
dataset=officehome
model=vit
method=fedavg
clients=2
rounds=1
generated=1
```

确认通信和训练闭环成功后，再逐步扩大：

```text
clients: 2 -> 4 -> 8
rounds: 1 -> 2 -> 5
generated: 1 -> 5 -> 20
method: fedavg -> fedprox/fedopt/fedproto/moon
```

不要一开始直接跑完整矩阵，否则失败时很难区分是网络、数据、模型权重还是训练逻辑问题。
