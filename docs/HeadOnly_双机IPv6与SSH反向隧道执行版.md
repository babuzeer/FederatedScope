# HeadOnly 双机 IPv6 与 SSH 反向隧道执行版

本文按你当前两台机器的地址编写。

```text
服务器 IPv6:
2001:da8:215:6a01:be24:11ff:fe58:570d

本地电脑稳定 IPv6:
2001:da8:215:3c0a:f51:4d71:80:8075

本地电脑临时 IPv6:
2001:da8:215:3c0a:b16e:b5cc:8f27:d60

服务器 IPv4:
10.112.81.135

本地电脑 IPv4:
10.129.222.189
```

优先使用 IPv6。不要使用临时 IPv6。IPv4 都是 `10.x` 私网地址，暂时不作为首选。

## 0. 约定

把 `<SERVER_SSH>` 替换成你能 SSH 到服务器的地址，例如：

```text
user@2001:da8:215:6a01:be24:11ff:fe58:570d
```

如果 SSH IPv6 需要方括号，OpenSSH 常见写法是：

```powershell
ssh user@2001:da8:215:6a01:be24:11ff:fe58:570d
```

如果你的 SSH 已经有别名，例如 `seetacloud`，后续直接用别名。

端口约定：

```text
server_port = 51251
client_1_port = 52251
client_2_port = 52252
```

## 1. 测试本地电脑能访问服务器

### 1.1 服务器开临时监听

服务器执行：

```bash
python3 -m http.server 51251 --bind ::
```

保持这个窗口不要关。

### 1.2 本地 PowerShell 测服务器端口

本地电脑 PowerShell 执行：

```powershell
Test-NetConnection 2001:da8:215:6a01:be24:11ff:fe58:570d -Port 51251
```

成功标准：

```text
TcpTestSucceeded : True
```

也可以用 Python 测：

```powershell
python -c "import socket; socket.create_connection(('2001:da8:215:6a01:be24:11ff:fe58:570d', 51251), 5); print('ok')"
```

成功后，在服务器监听窗口按 `Ctrl+C` 退出。

## 2. 测试服务器能访问本地电脑

先不改防火墙规则，因为你当前 PowerShell 没有管理员权限。

### 2.1 本地 PowerShell 开 IPv6 监听

本地电脑 PowerShell 执行：

```powershell
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::IPv6Any, 52251)
$listener.Start()
"listening on 52251"
$client = $listener.AcceptTcpClient()
"connected"
$client.Close()
$listener.Stop()
```

这个窗口会卡在 `AcceptTcpClient()`，这是正常的，表示正在等待服务器连接。

### 2.2 服务器连接本地电脑

服务器执行：

```bash
python3 -c "import socket; socket.create_connection(('2001:da8:215:3c0a:f51:4d71:80:8075', 52251), 5); print('ok')"
```

成功标准：

服务器输出：

```text
ok
```

本地 PowerShell 输出：

```text
connected
```

如果成功，跳到第 4 节，直接用 IPv6 直连配置。

如果失败，继续第 3 节，使用 SSH 反向隧道。

## 3. 无管理员权限时使用 SSH 反向隧道

这个方案不要求服务器直接访问本地电脑入站端口。逻辑是：

```text
server 连接 server 自己的 127.0.0.1:52251
  -> SSH 反向隧道
  -> 转发到本地电脑 127.0.0.1:52251
```

FederatedScope 里 client 上报给 server 的地址写成 `127.0.0.1:52251`，server 实际连的是服务器本机上的反向隧道端口。

### 3.1 本地电脑确认能 SSH 到服务器

本地 PowerShell：

```powershell
ssh <SERVER_SSH> "hostname; date"
```

成功后继续。

### 3.2 本地电脑先开一个本地测试监听

本地 PowerShell 窗口 A：

```powershell
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 52251)
$listener.Start()
"local listening on 127.0.0.1:52251"
$client = $listener.AcceptTcpClient()
"connected through reverse tunnel"
$client.Close()
$listener.Stop()
```

### 3.3 本地电脑打开 SSH 反向隧道

本地 PowerShell 窗口 B：

```powershell
ssh -N -R 52251:127.0.0.1:52251 <SERVER_SSH>
```

这个窗口会保持不退出，这是正常的。

如果提示远端端口已占用，换端口或先在服务器查占用：

```bash
ss -ltnp | grep 52251 || true
```

### 3.4 服务器测试反向隧道

服务器执行：

```bash
python3 -c "import socket; socket.create_connection(('127.0.0.1', 52251), 5); print('ok')"
```

成功标准：

服务器输出：

```text
ok
```

本地 PowerShell 窗口 A 输出：

```text
connected through reverse tunnel
```

### 3.5 为两个 client 建两个反向隧道

后续 smoke test 用 2 个 client，因此本地电脑 PowerShell 新开一个窗口，执行：

```powershell
ssh -N `
  -R 52251:127.0.0.1:52251 `
  -R 52252:127.0.0.1:52252 `
  <SERVER_SSH>
```

保持这个窗口一直开着。训练结束前不要关闭。

## 4. 生成 HeadOnly smoke test 配置

以下给两套配置。二选一。

## 4A. IPv6 直连配置

如果第 2 节成功，服务器可以直连本地电脑，使用这一套。

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

/root/miniconda3/envs/fs/bin/python \
  scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id two_machine_ipv6_smoke \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --head-only \
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
  --feature-cache-version two_machine_ipv6_smoke_v1
```

case 目录：

```text
scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly
```

## 4B. SSH 反向隧道配置

如果第 2 节失败，但第 3 节反向隧道成功，使用这一套。

注意：这里 `--client-host` 故意写 `127.0.0.1`。这是让 server 连接服务器本机的反向隧道端口。

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

/root/miniconda3/envs/fs/bin/python \
  scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --run-id two_machine_tunnel_smoke \
  --dataset officehome \
  --model vit \
  --method fedavg \
  --head-only \
  --server-host '[2001:da8:215:6a01:be24:11ff:fe58:570d]' \
  --server-bind-host '[::]' \
  --client-host '127.0.0.1' \
  --server-port 51251 \
  --client-port-base 52250 \
  --clients 2 \
  --sample-clients 2 \
  --rounds 1 \
  --num-generated-per-sample 1 \
  --num-generated-per-prototype 1 \
  --target-size-per-class 1 \
  --feature-cache-version two_machine_tunnel_smoke_v1
```

case 目录：

```text
scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_tunnel_smoke/officehome_vit_fedavg_headonly
```

## 5. 同步 case 到本地电脑

如果本地电脑也有同一路径的 WSL 仓库：

本地 WSL 或 Git Bash 执行：

```bash
scp -r <SERVER_SSH>:/root/autodl-tmp/FederatedScope/scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke \
  /root/autodl-tmp/FederatedScope/scripts/distributed_scripts/ggeur_multimachine/runs/
```

如果你用的是隧道配置，把目录名换成：

```bash
two_machine_tunnel_smoke
```

也可以直接同步整个仓库，但不要覆盖你本地未保存的修改。

## 6. 启动训练

下面命令以 IPv6 直连 case 为例。如果你使用隧道 case，把路径里的 `two_machine_ipv6_smoke` 改成 `two_machine_tunnel_smoke`。

### 6.1 服务器启动 server

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly
```

看日志：

```bash
tail -f scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

需要看到：

```text
Listen to
```

### 6.2 本地电脑启动 clients

本地 WSL/Linux 执行：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly
```

如果使用隧道配置：

```bash
cd /root/autodl-tmp/FederatedScope

PYTHON_BIN=/root/miniconda3/envs/fs/bin/python \
CLIENT_START_GAP=2 \
bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_tunnel_smoke/officehome_vit_fedavg_headonly
```

## 7. 成功判定

服务器日志：

```bash
grep -E "Client #[0-9]+ has joined|Received statistics quorum|Broadcasting global covariances|Broadcasting round|Training finished" \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

隧道 case 改路径：

```bash
grep -E "Client #[0-9]+ has joined|Received statistics quorum|Broadcasting global covariances|Broadcasting round|Training finished" \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_tunnel_smoke/officehome_vit_fedavg_headonly/logs/server.log
```

成功时应看到至少：

```text
Server: Client #1 has joined in
Server: Client #2 has joined in
Server: Training finished
```

## 8. 停止残留进程

服务器执行：

```bash
cd /root/autodl-tmp/FederatedScope

bash scripts/distributed_scripts/ggeur_multimachine/stop_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly
```

本地电脑 WSL/Linux 也执行一次：

```bash
cd /root/autodl-tmp/FederatedScope

bash scripts/distributed_scripts/ggeur_multimachine/stop_case.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/two_machine_ipv6_smoke/officehome_vit_fedavg_headonly
```

如果使用隧道 case，把路径换成 `two_machine_tunnel_smoke`。

SSH 反向隧道窗口按 `Ctrl+C` 关闭。

## 9. 常见错误处理

### 9.1 `New-NetFirewallRule : 拒绝访问`

当前 PowerShell 不是管理员权限。可以不用这条命令，先走第 2 节直连测试；如果失败，走第 3 节 SSH 反向隧道。

### 9.2 server 日志一直等 client join

本地 client 没连上 server。检查：

```powershell
python -c "import socket; socket.create_connection(('2001:da8:215:6a01:be24:11ff:fe58:570d', 51251), 5); print('ok')"
```

### 9.3 client join 了，但后续卡住

server 连不上 client。IPv6 直连方案回到第 2 节；隧道方案检查反向隧道窗口是否还开着。

服务器检查反向隧道端口：

```bash
ss -ltnp | grep -E '52251|52252' || true
```

### 9.4 gRPC IPv6 地址报错

生成配置时 IPv6 地址必须带方括号：

```text
[2001:da8:215:6a01:be24:11ff:fe58:570d]
[2001:da8:215:3c0a:f51:4d71:80:8075]
[::]
```

不要写成裸 IPv6：

```text
2001:da8:215:...
```
