"""Запуск настоящего start.sh с подменёнными процессами: без сервера и скачиваний."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SETTING_NAMES = (
    "USE_LLM", "LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
    "STREAMLIT_SERVER_PORT", "ENGINEERING_DATA_DIR", "SENTENCE_TRANSFORMERS_HOME",
)


class StartupRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="engineering-startup-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "core").mkdir()
        (self.root / "bin").mkdir()
        shutil.copy2(PROJECT / "start.sh", self.root / "start.sh")
        shutil.copy2(PROJECT / "core/config.py", self.root / "core/config.py")
        self.trace = self.root / "trace.jsonl"
        self.environment = {key: value for key, value in os.environ.items() if key not in SETTING_NAMES}
        self.environment.update({
            "PATH": f"{self.root / 'bin'}:/usr/bin:/bin",
            "STARTUP_TEST_TRACE": str(self.trace),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        self._write_executable(self.root / "bin/python", self._python_wrapper("fallback"))
        self._write_executable(self.root / "bin/curl", '''
record("curl", sys.argv[1:])
sys.exit(int(os.environ.get("STARTUP_TEST_CURL_EXIT", "0")))
''')
        self._write_executable(self.root / "bin/ollama", '''
record("ollama", sys.argv[1:])
sys.exit(int(os.environ.get("STARTUP_TEST_OLLAMA_EXIT", "0")))
''')

    def _write_executable(self, path: Path, body: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\n" + '''
import json
import os
import sys
def record(command, arguments):
    with open(os.environ["STARTUP_TEST_TRACE"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "command": command,
            "arguments": arguments,
            "model": os.environ.get("OLLAMA_MODEL"),
            "host": os.environ.get("OLLAMA_HOST"),
        }) + "\\n")
''' + body, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def _python_wrapper(name: str) -> str:
        return f'''
record({name!r}, sys.argv[1:])
if sys.argv[1:3] == ["-m", "streamlit"]:
    sys.exit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
'''

    def _run_start(self, settings: dict[str, str] | None = None):
        result = subprocess.run(
            ["/bin/bash", str(self.root / "start.sh")],
            cwd=self.root.parent,
            env={**self.environment, **(settings or {})},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        records = [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[-1]["arguments"][:4], ["-m", "streamlit", "run", "app.py"])
        return records

    def _read_config(self, settings: dict[str, str] | None = None):
        result = subprocess.run(
            [sys.executable, "-c", '''
import json
import os
import runpy
import sys
config = runpy.run_path(sys.argv[1])
print(json.dumps({
    "data": str(config["DATA_DIR"]),
    "raw": str(config["RAW_DIR"]),
    "cache": os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
    "model": config["OLLAMA_MODEL"],
}))
''', str(self.root / "core/config.py")],
            cwd=self.root.parent,
            env={**self.environment, **(settings or {})},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_existing_model_from_op_env_and_project_python_are_used(self):
        (self.root / "op.env").write_text("OLLAMA_MODEL=llama3.1:8b\nSTREAMLIT_SERVER_PORT=8502\n", encoding="utf-8")
        self._write_executable(self.root / ".venv/bin/python", self._python_wrapper("venv"))
        records = self._run_start()
        self.assertEqual(records[0]["command"], "venv")
        self.assertEqual(records[-1]["command"], "venv")
        self.assertIn("--server.port=8502", records[-1]["arguments"])
        model_calls = [item["arguments"] for item in records if item["command"] == "ollama"]
        self.assertEqual(model_calls, [["show", "llama3.1:8b"]])
        self.assertEqual(records[-1]["model"], self._read_config()["model"])

    def test_environment_then_dotenv_then_op_env_priority_matches_python(self):
        (self.root / "op.env").write_text("OLLAMA_MODEL=op-model\n", encoding="utf-8")
        (self.root / ".env").write_text("OLLAMA_MODEL=dotenv-model\n", encoding="utf-8")
        for settings, expected in (({}, "dotenv-model"), ({"OLLAMA_MODEL": "environment-model"}, "environment-model")):
            with self.subTest(expected=expected):
                records = self._run_start(settings)
                self.assertEqual(records[-1]["model"], expected)
                self.assertEqual(self._read_config(settings)["model"], expected)

    def test_disabled_llm_and_none_provider_skip_ollama(self):
        for setting in ("USE_LLM=FALSE", "LLM_PROVIDER=NONE"):
            with self.subTest(setting=setting):
                (self.root / ".env").write_text(setting + "\n", encoding="utf-8")
                self.trace.unlink(missing_ok=True)
                records = self._run_start()
                self.assertEqual([item["command"] for item in records], ["fallback", "fallback"])

    def test_unavailable_remote_ollama_still_launches_streamlit(self):
        records = self._run_start({
            "OLLAMA_BASE_URL": "http://unavailable.example:11434/",
            "STARTUP_TEST_CURL_EXIT": "7",
        })
        self.assertFalse(any(item["command"] == "ollama" for item in records))
        self.assertTrue(all(
            item["arguments"][-1] == "http://unavailable.example:11434/api/tags"
            for item in records if item["command"] == "curl"
        ))

    def test_failed_download_uses_configured_model_and_keeps_ui(self):
        records = self._run_start({"OLLAMA_MODEL": "requested-model", "STARTUP_TEST_OLLAMA_EXIT": "1"})
        model_calls = [item for item in records if item["command"] == "ollama"]
        self.assertEqual([item["arguments"] for item in model_calls], [
            ["show", "requested-model"], ["pull", "requested-model"],
        ])
        self.assertTrue(all(item["host"] == "http://localhost:11434" for item in model_calls))

    def test_dotenv_values_are_not_executed_as_shell_code(self):
        marker = self.root / "must-not-exist"
        model = f"literal-$(touch {marker})-`touch {marker}`; echo unexpected"
        (self.root / ".env").write_text(f"OLLAMA_MODEL='{model}'\n", encoding="utf-8")
        records = self._run_start()
        self.assertEqual(records[-1]["model"], model)
        self.assertFalse(marker.exists())

    def test_multiline_startup_values_cannot_inject_another_setting(self):
        result = subprocess.run(
            ["/bin/bash", str(self.root / "start.sh")], cwd=self.root.parent,
            env={**self.environment, "OLLAMA_MODEL": "model\nUSE_LLM=false"},
            capture_output=True, text=True, timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OLLAMA_MODEL", result.stderr)
        records = [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)

    def test_relative_data_and_embedding_paths_use_project_root(self):
        (self.root / ".env").write_text(
            "ENGINEERING_DATA_DIR=local-data\nSENTENCE_TRANSFORMERS_HOME=local-cache/embeddings\n",
            encoding="utf-8",
        )
        config = self._read_config()
        self.assertEqual(config["data"], str(self.root / "local-data"))
        self.assertEqual(config["raw"], str(self.root / "local-data/raw"))
        self.assertEqual(config["cache"], str(self.root / "local-cache/embeddings"))
        override = self._read_config({"SENTENCE_TRANSFORMERS_HOME": str(self.root / "override")})
        self.assertEqual(override["cache"], str(self.root / "override"))

    def test_existing_project_embedding_cache_is_used_without_hiding_hf_cache(self):
        self.assertIsNone(self._read_config()["cache"])
        (self.root / "data/embeddings").mkdir()
        self.assertEqual(self._read_config()["cache"], str(self.root / "data/embeddings"))


if __name__ == "__main__":
    unittest.main()

# ИСПРАВЛЕНО: регрессии .env/op.env, выбора Python, кэша, отказа Ollama и безопасного чтения настроек.
