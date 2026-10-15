"""The site configuration and the metadata-only credential checks."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

from tritonchip_agent_harness.provider import HARDENING, AgentContainer, Mount  # noqa: E402
from tritonchip_agent_harness.siteconfig import SiteError, credential_status, load_site  # noqa: E402

EXAMPLE = HERE / "examples" / "site.json"


class SiteTest(unittest.TestCase):
    def write(self, value) -> Path:
        d = Path(tempfile.mkdtemp())
        p = d / "site.json"
        p.write_text(json.dumps(value))
        return p

    def test_example_loads_and_names_no_credential(self):
        site = load_site(EXAMPLE)
        self.assertEqual(set(site.accounts), {"claude-subscription", "codex-subscription", "glm-coding-plan"})
        text = EXAMPLE.read_text()
        for word in ("sk-ant", "token\":", "api_key\":", "password"):
            self.assertNotIn(word, text)

    def test_strictness(self):
        base = json.loads(EXAMPLE.read_text())
        for mutate, message in [
            (lambda v: v.update(extra=1), "unknown"),
            (lambda v: v["network"]["routes"].update({"api.anthropic.com": "upstream"}), "upstream_proxy"),
            (lambda v: v["accounts"][0].update(label="Bad Label"), "label"),
            (lambda v: v["accounts"][0].update(hosts=["unrouted.example.com"]), "hosts"),
            (lambda v: v["network"].update(upstream_proxy="http://user:pw@127.0.0.1:1"), "credentials"),
        ]:
            v = json.loads(json.dumps(base))
            mutate(v)
            with self.assertRaisesRegex(SiteError, message):
                load_site(self.write(v))

    def test_credential_status_reads_shape_not_values(self):
        d = Path(tempfile.mkdtemp())
        env = d / "claude.env"
        env.write_text("CLAUDE_CODE_OAUTH_TOKEN=abcdefgh\n")
        os.chmod(env, 0o600)
        v = json.loads(EXAMPLE.read_text())
        v["accounts"] = [dict(v["accounts"][0], credential=str(env))]
        site = load_site(self.write(v))
        status = credential_status(site.accounts["claude-subscription"])
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["variables"], {"CLAUDE_CODE_OAUTH_TOKEN": 8})
        self.assertNotIn("abcdefgh", json.dumps(status))
        os.chmod(env, 0o644)
        self.assertFalse(credential_status(site.accounts["claude-subscription"])["ok"])


class ProviderTest(unittest.TestCase):
    def test_hardening_is_always_there_and_values_stay_off_the_command_line(self):
        c = AgentContainer(name="n", image="img", cpus=2, memory="4g", pids=512,
                           mounts=[Mount(Path("/x"), "/work", read_only=False)],
                           env={"A": "1"}, env_files=[Path("/secret.env")])
        args = c.run_args(["true"], detach=True, remove=False)
        joined = " ".join(args)
        for flag in HARDENING:
            self.assertIn(flag, args)
        self.assertIn("--env-file /secret.env", joined)
        self.assertIn("--memory-swap 4g", joined)
        self.assertIn("uid=1000,gid=1000", joined)
        self.assertEqual(args[-2:], ["img", "true"])


if __name__ == "__main__":
    unittest.main()
