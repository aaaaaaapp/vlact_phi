# 在本地浏览器打开服务器上的 Tokenizer HTML 报告

本说明记录 2026-10-26 已实际验证的做法。报告文件是 `docs/ROBOTWIN_TOKENIZER_RESULTS_SNAPSHOT.html`；网页服务使用仓库脚本 `scripts/physical_tokens/serve_tokenizer_results.py`，基于 Python 标准库，不需要安装 Web 框架。**页面是固定时间点快照，实验继续运行不会自动改写其中的数字。**

## 当前访问方式

服务运行在 node1（`10.120.154.139`）的 8765 端口。浏览器打开：

<http://10.120.154.139:8765/>

根路径会跳转到 `/docs/ROBOTWIN_TOKENIZER_RESULTS_SNAPSHOT.html`。已从 node1 和 node2 验证页面返回 HTTP 200，用户也确认可以从本地浏览器访问。

2026-10-28 另在 node001 的 **8767** 端口启动了同一白名单服务，以便打开自回归训练页。内网可直接访问：<http://10.120.154.139:8767/docs/ROBOTWIN_AR_GENERATION_TRAINING.html>。服务器端通过内网地址请求已确认返回 HTTP 200。`localhost` 始终指浏览器所在电脑；若要使用 `http://localhost:8767/docs/ROBOTWIN_AR_GENERATION_TRAINING.html`，必须先在**本地电脑**终端保持以下隧道运行：

```bash
ssh -N -L 8767:127.0.0.1:8767 liujn@10.120.154.139
```

若出现 `ERR_EMPTY_RESPONSE`，先试内网直连地址；能直连则检查本地 8767 是否被其他程序占用、SSH 隧道是否确实连到 node001。服务日志为 `artifacts/robotwin_full50_study/tokenizer_results_http_8767.log`。

## 重新启动服务

先登录 node1，并进入本仓库：

```bash
cd /mnt/jfzn/liujn/code/VLAct-PhysToken
ss -ltn | grep ':8765'
```

若端口已有服务，先检查进程，不要重复启动。若端口空闲，运行：

```bash
setsid -f /mnt/jfzn/miniconda3/envs/starvla_qwen/bin/python -u \
  scripts/physical_tokens/serve_tokenizer_results.py \
  --host 0.0.0.0 --port 8765 \
  > artifacts/robotwin_full50_study/tokenizer_results_http.log 2>&1 < /dev/null
```

`setsid -f` 让服务脱离当前终端会话；本次实践中只用 `nohup ... &` 的首次启动没有留下进程，因此采用上面的命令。`--host 0.0.0.0` 使集群内网地址可直连；脚本默认绑定 `127.0.0.1`，需要只允许 SSH 转发访问时可省略 `--host`。日志写在 `artifacts/robotwin_full50_study/tokenizer_results_http.log`。

检查服务：

```bash
ss -ltn | grep ':8765'
curl -I http://127.0.0.1:8765/
curl -I http://127.0.0.1:8765/docs/ROBOTWIN_TOKENIZER_RESULTS_SNAPSHOT.html
```

预期根路径返回 `302`，HTML 路径返回 `200`。若报 `Address already in use`，表示 8765 已被占用，应先识别现有服务，而不是直接启动第二份。

## 本地电脑不能直连时

在**本地电脑**的终端执行并保持连接：

```bash
ssh -N -L 8765:127.0.0.1:8765 liujn@10.120.154.139
```

然后在本地浏览器打开 <http://127.0.0.1:8765/>。若本地 8765 已被占用，可把命令左侧端口改为 8766，并访问 `http://127.0.0.1:8766/`。

## 访问范围与维护

服务脚本采用**明确白名单**：只提供这份 HTML、对应的实验 Markdown 和 HTML 中已列出的少量原始评估 JSON；其他路径返回 404，不提供目录浏览，也不暴露整个仓库。若以后给报告新增链接，需要同时在 `serve_tokenizer_results.py` 的 `ALLOWED` 列表中加入目标文件，否则浏览器点击会得到 404。绑定 `0.0.0.0` 仅适合可信的内网；若要跨公网访问，应改用 `127.0.0.1` 加 SSH 转发，或另加认证。

要停止服务，先查进程并确认它确实是该脚本，再终止**具体 PID**：

```bash
ps -eo pid,etime,args | grep '[s]erve_tokenizer_results.py'
kill -TERM 12345  # 把 12345 换成上一行确认后的 PID
```

不要使用会误伤其他服务的宽泛 `pkill python`。改动服务脚本后需按上述方式停旧进程、重启；仅修改 HTML 文件时刷新浏览器即可（服务对文件设置 `Cache-Control: no-store`）。
