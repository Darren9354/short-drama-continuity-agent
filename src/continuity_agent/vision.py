"""Standalone image-capable backends; no dependency on a parent application."""
from __future__ import annotations

import base64
import math
import os
from pathlib import Path
import subprocess
import tempfile


class LLMClient:
    def __init__(self, *, provider_name=None, model_vision=None, codex_chat_only=True):
        if not codex_chat_only:
            raise ValueError("视觉质检仅支持只读模型执行")
        self.provider_name = provider_name or os.environ.get("CONTINUITY_PROVIDER", "codex")
        if self.provider_name not in {"codex", "openai-compatible"}:
            raise ValueError("视觉后端应为codex或openai-compatible")
        self.command = os.environ.get("CODEX_COMMAND", "codex")
        self.timeout_seconds = float(os.environ.get("CONTINUITY_TIMEOUT_SECONDS", "600"))
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("模型调用超时须为正的有限秒数")
        self.model_vision = model_vision or os.environ.get(
            "CODEX_MODEL" if self.provider_name == "codex" else "VISION_MODEL"
        ) or ("codex" if self.provider_name == "codex" else "")
        self._client = None
        if self.provider_name == "openai-compatible":
            key = os.environ.get("VISION_API_KEY")
            if not key or not self.model_vision:
                raise ValueError("API后端需要VISION_API_KEY和支持图片的VISION_MODEL")
            try:
                from openai import OpenAI
            except ModuleNotFoundError:
                raise ValueError('API后端需先安装：python -m pip install ".[api]"') from None
            self._client = OpenAI(
                api_key=key,
                base_url=os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1"),
                timeout=self.timeout_seconds,
                max_retries=0,
            )

    @staticmethod
    def _prompt(messages):
        lines = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(str(p.get("text", "")) for p in content
                                 if isinstance(p, dict) and p.get("type") == "text")
            else:
                raise ValueError("模型消息内容须为文字或图文数组")
            if text.strip():
                lines.append("[" + str(message.get("role", "user")).upper() + "]\n" + text)
        return "\n\n".join(lines)

    def chat(self, messages, *, model=None):
        selected = model or self.model_vision
        if self.provider_name != "codex":
            result = self._client.chat.completions.create(model=selected, messages=messages)
            text = result.choices[0].message.content if result.choices else None
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Vision API returned an empty response")
            return text.strip()
        with tempfile.TemporaryDirectory(prefix="continuity-codex-") as directory:
            output = Path(directory) / "response.txt"
            command = [self.command, "exec", "--sandbox", "read-only", "--skip-git-repo-check",
                       "--output-last-message", str(output), "--ignore-user-config"]
            if selected and selected not in {"codex", "default"}:
                command.extend(["--model", selected])
            for feature in ("shell_tool", "apps", "browser_use", "browser_use_external",
                            "in_app_browser", "multi_agent", "view_image"):
                command.extend(["--disable", feature])
            command.extend(["--config", 'web_search="disabled"'])
            index = 0
            for message in messages:
                content = message.get("content", "")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image_url":
                        continue
                    url = str((part.get("image_url") or {}).get("url", ""))
                    header, separator, data = url.partition(",")
                    if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64"}:
                        raise ValueError("Codex image input requires an embedded PNG or JPEG")
                    extension = "png" if "png" in header else "jpg"
                    image = Path(directory) / f"frame-{index}.{extension}"
                    image.write_bytes(base64.b64decode(data, validate=True))
                    image.chmod(0o600)
                    command.extend(["--image", str(image)])
                    index += 1
            command.append("-")
            try:
                result = subprocess.run(command, input=self._prompt(messages), capture_output=True,
                                        text=True, check=False, timeout=self.timeout_seconds,
                                        cwd=Path(directory), env=os.environ.copy())
            except subprocess.TimeoutExpired:
                raise RuntimeError("Codex vision runtime timed out; no automatic retry") from None
            if result.returncode:
                raise RuntimeError(f"Codex vision runtime failed (exit {result.returncode})")
            if not output.is_file() or not output.read_text(encoding="utf-8").strip():
                raise RuntimeError("Codex vision runtime returned an empty response")
            return output.read_text(encoding="utf-8").strip()
