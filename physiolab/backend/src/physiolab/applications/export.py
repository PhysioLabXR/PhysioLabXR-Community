"""Marvin owns deployment adapters. This local HTML artifact is the first real target."""

import html
import json
from typing import Protocol

from physiolab.contracts import Project, Session


class DeploymentProvider(Protocol):
    def deploy(self, project: Project) -> str: ...


def export_preview(project: Project, session: Session) -> str:
    # Encode as data, never interpolate user intent into executable JavaScript.
    payload = json.dumps([i.model_dump() for i in session.inferences]).replace("<", "\\u003c")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(project.name)} — PhysioLab</title>
<style>body{{font:18px system-ui;background:#161826;color:#e9e9ed;margin:8vw}}
button{{font:inherit;padding:12px}}output{{display:block;font-size:64px;margin:32px 0}}</style>
<h1>{html.escape(project.name)}</h1><p>Recorded simulation · EMG gesture demo</p>
<p>{html.escape(project.prompt)}</p><output id="label">Ready</output><p id="value"></p>
<button id="play">Replay session</button><script>
const samples={payload};let timer;document.querySelector('#play').onclick=()=>{{
clearInterval(timer);let i=0;timer=setInterval(()=>{{
if(i>=samples.length){{clearInterval(timer);return;}}
const s=samples[i++];document.querySelector('#label').textContent=s.label;
document.querySelector('#value').textContent=s.value.toFixed(3)+' RMS';}},500);}};
</script></html>"""
