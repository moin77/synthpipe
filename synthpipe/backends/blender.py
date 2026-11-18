"""Run a Blender generator script in background mode and stream its output."""

import collections
import os
import subprocess
import sys


def build_command(cfg, run_config_path):
    """The exact command SynthPipe runs. Saved in metadata so anyone can rerun it."""
    b = cfg["blender"]
    return [
        b["exe"],
        "-b", b["blend_file"],          # background mode, open this .blend
        *b["extra_args"],
        "--python-exit-code", "1",      # without this Blender exits 0 even if the script crashes
        "-P", b["script"],
        "--",                           # everything after this goes to the script
        "--config", run_config_path,
    ]


def run(cfg, run_config_path, log_path, echo=True):
    """Start Blender, print its lines live, keep a full log on disk.

    Returns (return_code, last_lines) where last_lines are the final 40 lines,
    useful for showing why a run failed.
    """
    cmd = build_command(cfg, run_config_path)
    tail = collections.deque(maxlen=40)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        try:
            for line in proc.stdout:
                log.write(line)
                tail.append(line.rstrip("\n"))
                if echo and _worth_showing(line):
                    sys.stdout.write("  " + line)
                    sys.stdout.flush()
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            log.write("\nSTOPPED BY USER (Ctrl+C)\n")
            raise
    return proc.returncode, list(tail)


def _worth_showing(line):
    """Blender prints a lot of render noise. Show progress, warnings and errors only."""
    keep = ("[", "Found ", "Loaded config", "WARNING", "Error", "ERROR", "Traceback", "Done.", "Rendering on", "GPU setup")
    noise = ("Fra:", "| Time:", "Saved:")
    s = line.lstrip()
    if s.startswith(noise):
        return False
    return s.startswith(keep) or "Error" in s
