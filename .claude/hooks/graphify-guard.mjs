// PreToolUse hook: keeps Claude Code on the graphify knowledge graph when
// graphify-out/graph.json exists. Does nothing if the graph or the graphify CLI is missing.
// - Subagents (hook input has `agent_id`): blocks Bash/Grep/Glob/Read until that subagent
//   has run a `graphify` command, so every subagent starts from the graph. Reads inside
//   graphify-out/ are allowed.
// - Main session: forwards to `graphify hook-guard <mode>`, which only nudges.
// CLI lookup: $GRAPHIFY_BIN, the repo-local venv (.tools/graphify), then PATH.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const out = path.join(root, "graphify-out");
if (!fs.existsSync(path.join(out, "graph.json"))) process.exit(0);

const candidates = [
  process.env.GRAPHIFY_BIN,
  path.join(root, ".tools", "graphify", "Scripts", "graphify.exe"),
  path.join(root, ".tools", "graphify", "bin", "graphify"),
].filter(Boolean);
const local = candidates.find((p) => fs.existsSync(p));
const bin = local || "graphify";
if (!local && spawnSync("graphify", ["--help"], { stdio: "ignore" }).status !== 0) process.exit(0);

const mode = process.argv[2] === "read" ? "read" : "search";
let raw = "";
try { raw = fs.readFileSync(0, "utf8"); } catch {}
let input = {};
try { input = JSON.parse(raw || "{}"); } catch {}

if (input.agent_id) {
  const markers = path.join(out, ".subagents");
  const marker = path.join(markers, String(input.agent_id).replace(/[^\w.-]/g, "_"));
  const command = input.tool_input?.command || "";
  const target = input.tool_input?.file_path || input.tool_input?.path || "";
  if (input.tool_name === "Bash" && /\bgraphify\b/.test(command)) {
    try { fs.mkdirSync(markers, { recursive: true }); fs.writeFileSync(marker, ""); } catch {}
    process.exit(0);
  }
  if (fs.existsSync(marker)) process.exit(0);
  if (target && path.resolve(root, target).startsWith(out)) process.exit(0);
  const cli = local ? path.relative(root, local).replaceAll("\\", "/").replace(/\.exe$/, "") : "graphify";
  process.stdout.write(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason:
        `Subagents start from the graphify graph (AGENTS.md "Codebase context (graphify)"). ` +
        `Run \`${cli} query "<your task>"\` (or \`explain\` / \`path\`) via Bash first; ` +
        `this tool is unblocked after that.`,
    },
  }));
  process.exit(0);
}

const r = spawnSync(bin, ["hook-guard", mode], { input: raw, cwd: root, encoding: "utf8", timeout: 8000 });
if (r.status === 0 && r.stdout) process.stdout.write(r.stdout);
process.exit(0);
