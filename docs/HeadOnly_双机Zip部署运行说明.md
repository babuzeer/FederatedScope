# HeadOnly 双机 Zip 部署运行说明

本说明用于把本地已经准备好的 FederatedScope 代码打成 zip，通过
`scp` 上传到两台机器后运行最终 HeadOnly 60 客户端分布式训练。

## 机器口径

- server/4090: `10.112.81.135`
- client/8GB: `10.129.222.189`
- FederatedScope server 端口: `55051`
- FederatedScope client callback 端口: `56001-56060`

训练中的中心 server 跑在 4090 机器；8GB 机器跑 60 个 client 进程。
server 会反向连接 client callback 端口，所以两台机器必须互相可达。

## 本地打包

在本地 Windows 的项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\package_headonly_two_machine_bundle.ps1
```

输出示例：

```text
dist\FederatedScope_headonly_two_machine_YYYYMMDD_HHMMSS.zip
dist\FederatedScope_headonly_two_machine_YYYYMMDD_HHMMSS.zip.sha256
```

这个 zip 默认排除 `.git`、`exp`、数据集、缓存、日志和模型大文件。

## 上传到 4090 server

在可连接 `10.112.81.135` 的本地终端执行：

```powershell
scp dist\FederatedScope_headonly_two_machine_YYYYMMDD_HHMMSS.zip root@10.112.81.135:~/autodl-tmp/
```

登录 server 解压：

```bash
ssh root@10.112.81.135
mkdir -p ~/autodl-tmp/FederatedScope
cd ~/autodl-tmp/FederatedScope
unzip -o ~/autodl-tmp/FederatedScope_headonly_two_machine_YYYYMMDD_HHMMSS.zip
```

## 上传到 client 机器

如果 8GB Windows 已经能 SSH：

```powershell
scp dist\FederatedScope_headonly_two_machine_YYYYMMDD_HHMMSS.zip LKX@10.129.222.189:D:\FederatedScope_headonly_two_machine.zip
```

如果 Windows SSH 登录仍然不方便，可以通过远程桌面复制 zip，然后在
Windows 上解压到项目目录。

## 准备 60 客户端 case

推荐在 client 机器上生成 case，然后把同一个 case 目录同步到 server。
这样 60 个 client manifest 与 client 本地数据路径一致。

Linux/Git Bash:

```bash
cd /path/to/FederatedScope
SERVER_HOST=10.112.81.135 \
CLIENT_HOST=10.129.222.189 \
DATA_ROOT=/root/autodl-tmp/datasets/OfficeHomeDataset_10072016 \
bash scripts/distributed_scripts/ggeur_headonly_60c/prepare_case.sh
```

脚本输出最后一行是 case 目录，例如：

```text
scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

默认缓存目录为：

```text
exp/ggeur_headonly_real_cache/officehome_vitb16_60c_gen20_fcache_v1
```

代码会优先读取已有的 60 客户端 gen20 缓存；若缓存不存在或不兼容，才会重新生成。

## 启动 server

在 4090 server 上：

```bash
cd ~/autodl-tmp/FederatedScope
bash scripts/distributed_scripts/ggeur_headonly_60c/launch_server.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

## 启动 clients

在 client 机器上：

```bash
cd /path/to/FederatedScope
bash scripts/distributed_scripts/ggeur_headonly_60c/launch_clients.sh \
  scripts/distributed_scripts/ggeur_multimachine/runs/<RUN_ID>/officehome_vit_fedavg_headonly
```

## 日志

case 目录下查看：

```text
logs/server.log
logs/client_1.log
...
logs/client_60.log
system/server_run_info.log
system/clients_run_info.log
pids/*.pid
```

关键日志：

```text
Loaded augmented HeadOnly feature cache
HeadOnly cache-hot mode active
Round <n> aggregation complete
Training finished
```
