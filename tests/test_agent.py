from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from continuity_agent.__main__ import main
from continuity_agent.agent import (
    digest, evaluate, execute, finalize, load_manifest, prepare, read_json, review,
    review_scope, validate_analysis, windows, write_json,
)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    assert writer.isOpened()
    for index in range(30):
        image = np.zeros((120, 160, 3), np.uint8)
        image[:] = (0, 0, 230) if index < 15 else (0, 230, 0)
        cv2.rectangle(image, (index + 10, 30), (index + 25, 60), (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    return path


@pytest.fixture
def prepared(video, tmp_path):
    run = tmp_path / "run"
    manifest = prepare(video, run, interval=1, max_frames=30)
    return run, manifest


def response(ids, *, finding=False):
    return {"findings": [{"category": "appearance", "entity": "角色甲", "severity": "medium",
                           "confidence": 0.8, "description": "同一角色外套由红色变为绿色，需排除有意换装。",
                           "suggested_fix": "核对剧本时间线后统一外套颜色。",
                           "evidence_frame_ids": [ids[0], ids[-1]]}] if finding else [],
            "uncertainties": ["采样图无法验证全部中间动作。"], "state": "角色在画面内。", "anchors": [ids[0]]}


def records(manifest, *, finding=True):
    ids = [f["id"] for f in manifest["frames"]]
    return [{"frame_ids": ids, "result": response(ids, finding=finding)}]


def test_sample_cuts_timestamps_and_no_silent_overwrite(prepared, video):
    run, manifest = prepared
    assert manifest["status"] == "prepared"
    assert any(c["frame_index"] == 15 for c in manifest["cuts"])
    assert {14, 15, 16} <= {f["frame_index"] for f in manifest["frames"]}
    assert all(f["seconds"] == f["frame_index"] / 10 for f in manifest["frames"])
    assert manifest["source"]["sha256"] == digest(video)
    assert len(manifest["frames"]) == len({f["id"] for f in manifest["frames"]})
    with pytest.raises(FileExistsError):
        prepare(video, run)


def test_bounds_fail_closed_before_model(video, tmp_path):
    run = tmp_path / "too-many"
    with pytest.raises(ValueError, match="超过截图上限"):
        prepare(video, run, interval=0.1, max_frames=3)
    assert not (run / "manifest.json").exists()
    assert read_json(run / "preparation-error.json")["status"] == "failed"
    with pytest.raises(ValueError):
        prepare(video, tmp_path / "nan", interval=float("nan"))
    with pytest.raises(ValueError, match="时间范围无效"):
        prepare(video, tmp_path / "out-of-range", start=4)


def test_frame_tamper_is_detected(prepared):
    run, manifest = prepared
    path = run / manifest["frames"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="截图校验失败"):
        load_manifest(run)


@pytest.mark.parametrize("ids", [["a", "a"], ["a", "invented"]])
def test_hallucinated_or_duplicate_evidence_is_rejected(ids):
    with pytest.raises(ValueError):
        validate_analysis(response(ids, finding=True), {"a", "b"})


def test_partial_observations_cannot_claim_full_review(prepared):
    run, manifest = prepared
    ids = [f["id"] for f in manifest["frames"][:2]]
    report = finalize(run, [{"frame_ids": ids, "result": response(ids, finding=True)}], provenance={"kind": "test"})
    assert report["status"] == "partial_analysis"
    assert report["coverage"]["all_sampled_frames_analyzed"] is False
    assert report["findings"][0]["human_status"] == "pending"
    assert (run / "report.md").exists()


def test_live_loop_sends_real_images_carries_memory_and_records_usage(prepared):
    run, manifest = prepared
    calls = []

    def backend(messages):
        parts = messages[-1]["content"]
        context = json.loads(parts[0]["text"])
        ids = [f["id"] for f in context["frame_map"]]
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == len(ids)
        assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in images)
        if calls:
            assert context["previous_state"] == "角色在画面内。"
        calls.append(ids)
        return json.dumps(response(ids, finding=True), ensure_ascii=False)

    result = execute(run, backend=backend, batch_size=3, max_calls=20)
    assert len(calls) > 1
    assert result["status"] == "awaiting_human_review"
    assert result["coverage"]["analyzed_frames"] == len(manifest["frames"])
    status = read_json(run / "analysis/status.json")
    assert status["completed_calls"] == len(calls)
    assert status["cost_amount"] is None
    with pytest.raises(FileExistsError):
        execute(run, backend=backend)


def test_call_budget_is_checked_before_any_model_call(prepared):
    run, _ = prepared
    with pytest.raises(ValueError, match="超过--max-calls"):
        execute(run, backend=lambda _: pytest.fail("must not call"), batch_size=3, max_calls=1)
    assert not (run / "analysis").exists()


def test_model_failure_retains_partial_records_and_never_retries(prepared):
    run, _ = prepared
    calls = []

    def backend(messages):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated unknown response")
        context = json.loads(messages[-1]["content"][0]["text"])
        return response([f["id"] for f in context["frame_map"]])

    with pytest.raises(RuntimeError):
        execute(run, backend=backend, batch_size=3)
    assert len(calls) == 2
    assert (run / "analysis/batch-0000.json").exists()
    assert read_json(run / "analysis/status.json")["status"] == "failed"
    assert not (run / "report.json").exists()


def test_review_keeps_original_report_and_adds_audit_history(prepared):
    run, manifest = prepared
    report = finalize(run, records(manifest), provenance={"kind": "test"})
    original_hash = digest(run / "report.json")
    issue = report["findings"][0]["id"]
    review(run, issue, "confirmed", reviewer="测试复核人", note="已核对原始镜头")
    review(run, issue, "uncertain", reviewer="测试复核人", note="待确认是否跳时")
    assert len(read_json(run / "reviews.json")) == 2
    assert digest(run / "report.json") == original_hash
    assert "uncertain" in (run / "report.md").read_text()
    with pytest.raises(ValueError, match="找不到"):
        review(run, "invented", "confirmed", reviewer="x", note="x")


def test_evaluation_matches_truth_and_does_not_invent_rates(prepared):
    run, manifest = prepared
    report = finalize(run, records(manifest), provenance={"kind": "test"})
    found = report["findings"][0]
    truth = {"source_sha256": report["source"]["sha256"], "range": report["range"], "complete": True,
             "issues": [{k: found[k] for k in ("category", "entity", "start_seconds", "end_seconds")},
                        {"category": "action", "entity": "角色甲", "start_seconds": 1, "end_seconds": 2}]}
    result = evaluate(run, truth)
    assert result["per_category"]["appearance"]["precision"] == 1
    assert result["per_category"]["action"]["false_negative"] == 1
    assert result["per_category"]["prop_ownership"]["recall"] is None
    truth["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="哈希不一致"):
        evaluate(run, truth)


def test_artifact_path_traversal_rejected(prepared):
    run, manifest = prepared
    manifest["frames"][0]["path"] = "../outside.jpg"
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="越出"):
        load_manifest(run)


def test_cli_prepare_and_import_observations(video, tmp_path):
    run = tmp_path / "cli"
    assert main(["prepare", str(video), "--output", str(run)]) == 0
    data = tmp_path / "observations.json"
    write_json(data, records(read_json(run / "manifest.json")))
    assert main(["import-observations", str(run), str(data), "--observer", "test"]) == 0
    assert read_json(run / "report.json")["status"] == "awaiting_human_review"


def test_windows_have_overlap_without_empty_tail():
    frames = list(range(12))
    result = windows(frames, size=5, overlap=2)
    assert result == [[0, 1, 2, 3, 4], [3, 4, 5, 6, 7], [6, 7, 8, 9, 10], [9, 10, 11]]
    assert set().union(*map(set, result)) == set(frames)


def test_zero_finding_report_supports_scope_review(prepared):
    run, manifest = prepared
    finalize(run, records(manifest, finding=False), provenance={"kind": "test"})
    review_scope(run, "needs_more_evidence", reviewer="测试复核人", note="补手部特写")
    review_scope(run, "reviewed", reviewer="测试复核人", note="已逐帧核对本范围")
    assert len(read_json(run / "scope-reviews.json")) == 2


def test_unresolved_findings_block_scope_completion(prepared):
    run, manifest = prepared
    finalize(run, records(manifest), provenance={"kind": "test"})
    with pytest.raises(ValueError, match="仍有"):
        review_scope(run, "reviewed", reviewer="x", note="x")
    assert review_scope(run, "needs_more_evidence", reviewer="x", note="x")["decision"] == "needs_more_evidence"


def test_default_provider_adapter_uses_images_without_text_fallback(prepared, monkeypatch):
    from continuity_agent.vision import LLMClient
    run, _ = prepared
    seen = []

    def init(self, **kwargs):
        assert kwargs["codex_chat_only"] is True
        self.provider_name = "configured-provider"
        self.model_vision = "configured-vision-model"

    def chat(self, messages, *, model):
        assert model == "configured-vision-model"
        context = json.loads(messages[-1]["content"][0]["text"])
        seen.extend(p for p in messages[-1]["content"] if p["type"] == "image_url")
        return json.dumps(response([f["id"] for f in context["frame_map"]]))

    monkeypatch.setattr(LLMClient, "__init__", init)
    monkeypatch.setattr(LLMClient, "chat", chat)
    result = execute(run)
    assert seen
    assert result["provenance"]["model"] == "configured-vision-model"


def test_invalid_model_json_never_produces_clean_report(prepared):
    run, _ = prepared
    with pytest.raises(ValueError):
        execute(run, backend=lambda _: "not-json")
    assert read_json(run / "analysis/status.json")["attempted_calls"] == 1
    assert not (run / "report.json").exists()


def test_evaluation_uses_maximum_one_to_one_matching(prepared):
    run, manifest = prepared
    report = finalize(run, records(manifest), provenance={"kind": "test"})
    base = report["findings"][0]
    report["findings"] = [{**base, "start_seconds": 0, "end_seconds": 2},
                          {**base, "start_seconds": 0, "end_seconds": 0.5}]
    write_json(run / "report.json", report)
    truth = {"source_sha256": report["source"]["sha256"], "range": report["range"], "complete": True,
             "issues": [{"category": "appearance", "entity": base["entity"], "start_seconds": 0, "end_seconds": 0.5},
                        {"category": "appearance", "entity": base["entity"], "start_seconds": 1.5, "end_seconds": 2}]}
    assert evaluate(run, truth, tolerance=0)["per_category"]["appearance"]["true_positive"] == 2
