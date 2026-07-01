# HeadOnly hierarchical parameter-QPS case

This directory runs the current two-machine landing case:

- root server and one subserver are separate processes on the 4090 server
- 60 HeadOnly MLP clients are separate processes on the 8G client machine
- clients load existing feature caches, train MLP locally, upload parameters, then wait
- after root aggregation finishes, clients send real TCP read-param requests to the subserver
- the measured QPS is ACK-only parameter-read responses within one second after global ready

No training QPS is reported. The main metric is written as
`subserver_parameter_read_qps` in `subserver_1_events.jsonl` on the root server.

## Prepare on the 8G client machine

```powershell
cd D:\Projects\FederatedScope
.\scripts\distributed_scripts\headonly_hierarchical_qps\prepare_two_machine_case.ps1 -CheckCache
```

The command prints `case_dir=...`. Use that case directory in later commands.

If the 4090 repo copy is not already updated, create a small server bundle:

```powershell
.\scripts\distributed_scripts\headonly_hierarchical_qps\package_case_for_4090.ps1 -CaseDir <CASE_DIR>
```

Unzip the bundle under `/root/autodl-tmp/FederatedScope` on the 4090 server.

## Start on the 4090 server

```bash
cd /root/autodl-tmp/FederatedScope
export PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python
CASE_DIR=scripts/distributed_scripts/headonly_hierarchical_qps/runs/<RUN_ID>
bash scripts/distributed_scripts/headonly_hierarchical_qps/launch_root_server.sh "$CASE_DIR"
bash scripts/distributed_scripts/headonly_hierarchical_qps/launch_subserver.sh "$CASE_DIR"
```

Root and subserver logs are saved under:

```text
/root/autodl-tmp/FederatedScope/exp/headonly_hierarchical_qps/headonly60c-1sub-paramqps-YYYYMMDD
```

## Start on the 8G client machine

```powershell
cd D:\Projects\FederatedScope
.\scripts\distributed_scripts\headonly_hierarchical_qps\launch_clients.ps1 -CaseDir <CASE_DIR>
```

## One-command start from the 8G client machine

This uploads the server bundle to the 4090 server, starts root/subserver there,
then starts all clients on the 8G machine:

```powershell
cd D:\Projects\FederatedScope
.\scripts\distributed_scripts\headonly_hierarchical_qps\start_two_machine_from_8g.ps1 -Restart
```

## Status and stop

On 4090:

```bash
bash scripts/distributed_scripts/headonly_hierarchical_qps/status_case.sh "$CASE_DIR"
bash scripts/distributed_scripts/headonly_hierarchical_qps/stop_case.sh "$CASE_DIR"
```

On 8G:

```powershell
.\scripts\distributed_scripts\headonly_hierarchical_qps\status_clients.ps1 -CaseDir <CASE_DIR>
.\scripts\distributed_scripts\headonly_hierarchical_qps\stop_clients.ps1 -CaseDir <CASE_DIR> -Force
```
