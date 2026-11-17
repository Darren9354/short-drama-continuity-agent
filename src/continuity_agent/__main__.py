from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import evaluate, execute, finalize, prepare, read_json, review, review_scope, write_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="短剧连续性质检Agent：本地视频、截图证据及人工复核，无网页")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name, help="采样准备" if name == "prepare" else "采样并自动视觉质检")
        command.add_argument("video", type=Path)
        command.add_argument("--output", type=Path, required=True, help="新任务目录，不覆盖旧记录")
        command.add_argument("--baseline", type=Path, help="角色/道具基准JSON，可用已有series-bible.json")
        command.add_argument("--reference", type=Path, action="append", default=[], help="参考图，最多4张")
        command.add_argument("--interval", type=float, default=1.0)
        command.add_argument("--start", type=float, default=0)
        command.add_argument("--end", type=float)
        command.add_argument("--max-frames", type=int, default=240)
    for name in ("analyze",):
        command = commands.add_parser(name, help="对已准备任务自动视觉质检")
        command.add_argument("run", type=Path)
    for name in ("run", "analyze"):
        command = commands.choices[name]
        command.add_argument("--provider", help="codex或openai-compatible，默认CONTINUITY_PROVIDER/codex")
        command.add_argument("--model", help="可选视觉模型覆盖；不指定沿用配置")
        command.add_argument("--batch-size", type=int, default=12)
        command.add_argument("--max-calls", type=int, default=24)
    command = commands.add_parser("import-observations", help="导入本会话视觉检查记录，不调用外部模型")
    command.add_argument("run", type=Path)
    command.add_argument("observations", type=Path)
    command.add_argument("--observer", required=True)
    command = commands.add_parser("review", help="追加人工结论，不篡改模型证据")
    command.add_argument("run", type=Path)
    command.add_argument("issue_id")
    command.add_argument("--decision", choices=["confirmed", "dismissed", "uncertain"], required=True)
    command.add_argument("--reviewer", required=True)
    command.add_argument("--note", required=True)
    command = commands.add_parser("review-scope", help="确认整段报告已人工复核，或要求补证；零疑点时也适用")
    command.add_argument("run", type=Path)
    command.add_argument("--decision", choices=["reviewed", "needs_more_evidence"], required=True)
    command.add_argument("--reviewer", required=True)
    command.add_argument("--note", required=True)
    command = commands.add_parser("evaluate", help="与独立人工标注比较漏报/误报")
    command.add_argument("run", type=Path)
    command.add_argument("truth", type=Path)
    command.add_argument("--tolerance", type=float, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command in {"prepare", "run"}:
            manifest = prepare(args.video, args.output, baseline=args.baseline,
                               references=tuple(args.reference), interval=args.interval,
                               start=args.start, end=args.end, max_frames=args.max_frames)
            if args.command == "prepare":
                result = {"status": manifest["status"], "sampled_frames": len(manifest["frames"]),
                          "output": str(args.output)}
            else:
                result = execute(args.output, provider=args.provider, model=args.model,
                                 batch_size=args.batch_size, max_calls=args.max_calls)
        elif args.command == "analyze":
            result = execute(args.run, provider=args.provider, model=args.model,
                             batch_size=args.batch_size, max_calls=args.max_calls)
        elif args.command == "import-observations":
            result = finalize(args.run, read_json(args.observations),
                              provenance={"kind": "imported_visual_observations", "observer": args.observer,
                                          "model_calls": 0, "cost_amount": None,
                                          "cost_note": "本命令未调用模型；观察本身的成本不在本命令计量内。"})
        elif args.command == "review":
            result = review(args.run, args.issue_id, args.decision, reviewer=args.reviewer, note=args.note)
        elif args.command == "review-scope":
            result = review_scope(args.run, args.decision, reviewer=args.reviewer, note=args.note)
        else:
            result = evaluate(args.run, read_json(args.truth), tolerance=args.tolerance)
            write_json(args.run / "evaluation.json", result)
        if isinstance(result, dict) and "findings" in result:
            result = {"status": result["status"], "findings": len(result["findings"]), "coverage": result["coverage"]}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # Do not echo provider errors: they can contain URLs, auth headers or prompts.
        text = str(exc) if isinstance(exc, (ValueError, FileExistsError, FileNotFoundError)) else "处理失败，请检查本地analysis/status.json；未自动重试模型调用。"
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": text}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
