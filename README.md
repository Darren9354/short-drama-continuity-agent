# 短剧连续性质检 Agent v1

首版提供命令行视频质检，不启动网页或常驻服务。检查角色外观、道具归属和关键动作衔接，生成真实视频截图、时间点、修改建议、JSON及可本地打开的Markdown报告。所有疑点等待人工确认；没有发现不表示视频通过。

这是独立Python项目，从本项目根目录运行。安装及模型配置见下文。视频和角色基准仅在显式执行 `run`/`analyze` 时发送给配置的模型；`prepare`、导入观察和人工复核在本地完成。不会发布、通知微信或改动原视频。

## 安装与独立运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
codex login
.venv/bin/continuity-agent --help
```

默认后端为已安装并登录的Codex CLI，图像通过原生图片参数传入，运行目录为独立临时目录、只读沙箱，模型不获得Shell、浏览器或应用工具。项目无需导入或安装Enlightenment，也不读取其它项目的配置、工作区或微信账号。

配置使用环境变量，示例在 `.env.example`；该文件仅供参考，不会自动加载。常用变量是 `CODEX_COMMAND`、`CODEX_MODEL`、`CONTINUITY_TIMEOUT_SECONDS`。可选API后端先安装 `.[api]`，再设置 `CONTINUITY_PROVIDER=openai-compatible`、`VISION_API_KEY`、`VISION_MODEL` 和按需设置的 `VISION_BASE_URL`；必须选择支持图片的模型。API SDK禁用自动重试，失败不回退到纯文字判断。

角色与道具基准示例为 `examples/series-bible.json`。示例将检查产物保存在自行指定的新 `runs/` 子目录，报告和视频不会随源码提交。

```bash
# 自动分析：默认使用Codex CLI视觉输入
.venv/bin/python -m continuity_agent run \
  /path/to/episode.mp4 --baseline /path/to/series-bible.json \
  --output runs/episode-v1

# 只取证，不发起模型调用；可指定时间范围、采样密度及最多4张参考图
.venv/bin/python -m continuity_agent prepare \
  /path/to/episode.mp4 --start 10 --end 35 --interval 0.5 \
  --reference /path/to/character.png \
  --output runs/episode-preview

# 对已有取证目录分析；可选择codex或openai-compatible及支持图片的模型
.venv/bin/python -m continuity_agent analyze \
  runs/episode-preview --max-calls 24

# 人工复核，追加历史而不覆盖原始模型结论
.venv/bin/python -m continuity_agent review \
  runs/episode-v1 issue-实际ID \
  --decision confirmed --reviewer Reviewer --note '已核对前后镜头，确认衣服颜色变化'

# 整段报告复核；零疑点时也可使用。只代表完成复核，不表示视频无错误。
.venv/bin/python -m continuity_agent review-scope \
  runs/episode-v1 \
  --decision needs_more_evidence --reviewer Reviewer --note '请补充手部交接镜头'
```

默认每秒一张，使用HSV直方图差异识别疑似硬切并补取切镜前后帧；默认最多240张、24次视觉调用，每批12帧并重叠2帧。Agent将角色和道具状态、最多4张已见关键帧传到下一批，辅助跨镜头对比。可用 `--interval`、`--batch-size`、`--max-frames`、`--max-calls` 控制工作量。超过上限直接停止，不偷偷丢帧或裁短视频。最多检查10分钟范围，可用 `--start/--end` 分段。

每条疑点必须包含至少两张本批次看过的视频截图。模型不能自行编造时间点或引用未知图片；截图、基准及源视频都有SHA256记录。参考图不算视频时间证据。纯文字模型、接口失败、JSON格式或证据错误都停止，绝不退化为只读剧本的“通过”。调用之前记录attempted，保留已完成批次，不自动重试失败或未知调用；重做请保留原目录并使用新任务目录。

输出目录含 `manifest.json`、`baseline.json`、`frames/`、`contact-sheet-*.jpg`、`analysis/`、`observations.json`、`report.json`、`report.md`。`reviews.json`及`scope-reviews.json`保存人工逐项/整段复核历史；Markdown展示最新结论，原始report.json保持不变。处理时间、调用次数、图片数量可查；未取得模型账单时金额为null，不虚报免费或具体费用。默认使用Codex CLI只读图像输入；可选OpenAI兼容视觉API，不安装插件、购买额度或回退到纯文字检查。

也可由本会话查看截图后导入观察，明确标记来源，不冒充自动模型调用：

```bash
.venv/bin/python -m continuity_agent import-observations \
  /path/to/run /path/to/observations.json --observer '当前会话视觉检查'
```

观察文件为数组，每项含 `frame_ids` 和 `result`。`result`含 `findings`、`uncertainties`、`state`、`anchors`；findings每项含 `category`（appearance/prop_ownership/action）、`entity`、`severity`（low/medium/high）、`confidence`（0—1）、`description`、`suggested_fix`、`evidence_frame_ids`。缺少部分截图的观察报告固定标记partial_analysis。模型把握不是校准准确率。

独立人工标注后，可用以下命令评估。标注绑定源视频SHA256及检查范围；issue含category、entity、start_seconds、end_seconds，并设置complete表示该范围是否已完整标注。类别与对象名称精确匹配、时间段重叠（默认容差1秒）的一对一匹配用于统计TP/FP/FN、precision、recall。没有完整人工标注不能声称已测出漏报；没有负例总数，不输出FPR或总体准确率。

```bash
.venv/bin/python -m continuity_agent evaluate \
  /path/to/run /path/to/human-truth.json --tolerance 1
```

首版只接受可准确定位的恒定帧率视频，发现明显时间戳漂移即停止。动作检查针对截图中的姿态/方向/持物状态；不验证完整连续运动、口型、音频或物理仿真。遮挡、闪回、时间跳跃、光线与剪辑方式均需人工判断。短暂错误可能落在采样间隔之间。首版没有人工标注评测集，不声称检出率或商业可用准确率。


## 隐私与数据

仓库仅提供源码、空值环境变量示例和虚构测试数据。真实视频、参考图、运行报告、API密钥和本机配置应保留在本机；默认产物目录和常见凭据文件已列入 `.gitignore`。`prepare` 在本地取证，`run` / `analyze` 会把选定截图和基准发送给配置的模型后端。

## 许可证

本项目采用 [MIT License](LICENSE)。
