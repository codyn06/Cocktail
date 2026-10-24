// PreToolUse hook: makes every Claude Code subagent start from the graphify graph.
// Blocks a subagent's Bash/Grep/Glob/Read calls until that subagent has run a `graphify`
// command; reads inside graphify-out/ stay allowed. The main session is left to the
// official `graphify hook-guard` nudge (installed by `graphify claude install`).
// Does nothing when graphify-out/graph.json or the graphify CLI is missing.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const out = path.join(root, "graphify-out");
if (!fs.existsSync(path.join(out, "graph.json"))) process.exit(0);

let input = {};
try { input = JSON.parse(fs.readFileSync(0, "utf8") || "{}"); } catch {}
if (!input.agent_id) process.exit(0);
if (spawnSync("graphify", ["--help"], { stdio: "ignore" }).status !== 0) process.exit(0);

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

process.stdout.write(JSON.stringify({
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason:
      `Subagents start from the graphify graph (AGENTS.md "Codebase context (graphify)"). ` +
      `Run \`graphify query "<your task>"\` (or \`explain\` / \`path\`) via Bash first; ` +
      `this tool is unblocked after that.`,
  },
}));
process.exit(0);
