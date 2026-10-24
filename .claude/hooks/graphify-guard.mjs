// PreToolUse hook: forwards to `graphify hook-guard <mode>` so Claude Code is nudged
// toward graph queries when graphify-out/graph.json exists. Never blocks a tool call:
// if the graph or the graphify CLI is missing, it exits 0 silently.
// CLI lookup: $GRAPHIFY_BIN, the repo-local venv (.tools/graphify), then PATH.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
if (!fs.existsSync(path.join(root, "graphify-out", "graph.json"))) process.exit(0);

const candidates = [
  process.env.GRAPHIFY_BIN,
  path.join(root, ".tools", "graphify", "Scripts", "graphify.exe"),
  path.join(root, ".tools", "graphify", "bin", "graphify"),
].filter(Boolean);
const bin = candidates.find((p) => fs.existsSync(p)) || "graphify";

const mode = process.argv[2] === "read" ? "read" : "search";
let input = "";
try { input = fs.readFileSync(0, "utf8"); } catch {}
const r = spawnSync(bin, ["hook-guard", mode], { input, cwd: root, encoding: "utf8", timeout: 8000 });
if (r.status === 0 && r.stdout) process.stdout.write(r.stdout);
process.exit(0);
