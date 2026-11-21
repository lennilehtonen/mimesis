"""Static copy of the dashboard for GitHub Pages with one recorded take and its results.
    python scripts/build_site.py ep_173510   -> site/"""
import json, shutil, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent
ep = sys.argv[1] if len(sys.argv) > 1 else "ep_173510"
S, W, D = R / "site", R / "web", R / "data" / "web"
if S.exists():
    shutil.rmtree(S)
subprocess.run([sys.executable, str(R / "scripts" / "build_ui.py")], check=True)
shutil.copytree(W, S, ignore=shutil.ignore_patterns("screens", "shell.html"))
F = S / "files"; (F / "variants").mkdir(parents=True)
d = json.loads((D / f"ep_{ep}.json").read_text())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(D / f"ep_{ep}_overlay.mp4"), "-vf", "scale=1280:-2",
                "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-an", str(F / "overlay.mp4")], check=True)
shutil.copy(D / f"ep_{ep}_twin.mp4", F / "twin.mp4")
d["overlay"], d["twin"] = "files/overlay.mp4", "files/twin.mp4"
(F / "episode_0.json").write_text(json.dumps(d))
for p in (D / "variants").iterdir():
    shutil.copy(p, F / "variants" / p.name)
st = json.loads((D / "dataset_stats.json").read_text())
st["episodes"] = [{"n": 0, "id": ep, "task": d["task"], "duration": d["duration"]}]
st["episodes_real"] = 1
(F / "dataset_stats.json").write_text(json.dumps(st))
if (D / "metrics.json").exists():
    shutil.copy(D / "metrics.json", F / "metrics.json")
(S / ".nojekyll").write_text("")
print("site/ ->", sum(f.stat().st_size for f in S.rglob("*") if f.is_file()) // 1_000_000, "MB")
