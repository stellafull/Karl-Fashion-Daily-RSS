import subprocess


def test_plain_uv_run_python_can_import_deep_agents_package() -> None:
    completed = subprocess.run(
        ["uv", "run", "python", "-c", "from deep_agents.schemas import ResearchBrief; print(ResearchBrief.__name__)"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ResearchBrief"
