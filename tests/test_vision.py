"""The continuity backend must pass real image files to a read-only Codex run."""
import base64
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from continuity_agent.vision import LLMClient


@pytest.fixture
def client(tmp_path,monkeypatch):
    return LLMClient(provider_name='codex',codex_chat_only=True)


def test_images_are_attached_read_only_and_private_temporary_files_are_removed(client,monkeypatch):
    images=b'provided-frame-bytes';seen=[]
    def run(command,**kwargs):
        assert command[command.index('--sandbox')+1]=='read-only'
        assert '--ignore-user-config' in command and command[-1]=='-'
        assert 'web_search="disabled"' in command
        assert 'Inspect this frame.' in kwargs['input']
        assert 'base64,' not in kwargs['input']
        assert not any('Inspect this frame.' in item for item in command)
        assert not any('base64,' in item for item in command)
        assert kwargs['cwd']!=Path.cwd()
        image=Path(command[command.index('--image')+1]);assert image.read_bytes()==images
        output=Path(command[command.index('--output-last-message')+1]);output.write_text('{"findings":[]}')
        seen.extend((image,output,Path(kwargs['cwd'])))
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(subprocess,'run',run)
    messages=[dict(role='user',content=[dict(type='text',text='Inspect this frame.'),
        dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(images).decode()))])]
    assert client.chat(messages)=='{"findings":[]}'
    assert seen and all(not path.exists() for path in seen)


def test_unsupported_images_do_not_run_or_silently_become_text(client,monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('Invalid images must stop before model invocation'))
    with pytest.raises(ValueError,match='embedded PNG or JPEG'):
        client.chat([dict(role='user',content=[dict(type='image_url',image_url=dict(url='https://example.com/frame.jpg'))])])


@pytest.mark.parametrize('failure',['empty','timeout'])
def test_missing_model_result_is_not_a_success_and_temporary_directory_is_removed(client,monkeypatch,failure):
    paths=[]
    def run(command,**kwargs):
        output=Path(command[command.index('--output-last-message')+1]);paths.append(output.parent)
        if failure=='timeout':raise subprocess.TimeoutExpired(command,1)
        output.write_text('');return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(subprocess,'run',run)
    with pytest.raises(RuntimeError,match='empty response|timed out'):
        client.chat([dict(role='user',content='Inspect the evidence.')])
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.parametrize('failed',[False,True])
def test_api_keeps_images_disables_retry_and_never_falls_back_to_codex(monkeypatch,failed):
    monkeypatch.setenv('VISION_API_KEY','unit-test-key')
    monkeypatch.setenv('VISION_MODEL','unit-vision-model')
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('API failures must not switch providers'))
    seen=[]
    def create(**kwargs):
        seen.append(kwargs)
        if failed:raise RuntimeError('Provider unavailable')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"findings":[]}'))])
    def sdk(**kwargs):
        assert kwargs['max_retries']==0
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules,'openai',SimpleNamespace(OpenAI=sdk))
    backend=LLMClient(provider_name='openai-compatible')
    messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url='data:image/png;base64,dGVzdA=='))])]
    if failed:
        with pytest.raises(RuntimeError,match='Provider unavailable'):backend.chat(messages)
    else:assert backend.chat(messages)=='{"findings":[]}'
    assert seen==[dict(model='unit-vision-model',messages=messages)]


def test_api_requires_explicit_key_and_vision_model(monkeypatch):
    monkeypatch.delenv('VISION_API_KEY',raising=False)
    monkeypatch.delenv('VISION_MODEL',raising=False)
    with pytest.raises(ValueError,match='VISION_API_KEY'):
        LLMClient(provider_name='openai-compatible')
