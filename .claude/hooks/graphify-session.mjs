// SessionStart / SubagentStart hook: makes every Claude Code session and subagent start
// from the graphify knowledge graph. On a new session it builds or refreshes the graph
// (code-only `graphify update .`: local, no model calls) when it is missing or stale, then
// injects the graphify rules plus a compact excerpt of GRAPH_REPORT.md. Never fails the
// session: on any problem it exits 0, adding setup instructions instead of the excerpt.
// Usage: node graphify-session.mjs <SessionStart|SubagentStart>
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const event = process.argv[2] === "SubagentStart" ? "SubagentStart" : "SessionStart";
const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const out = path.join(root, "graphify-out");
const report = path.join(out, "GRAPH_REPORT.md");
const MAX = 8000; // additionalContext is capped at 10,000 characters per string

let input = {};
try { input = JSON.parse(fs.readFileSync(0, "utf8") || "{}"); } catch {}

const winBin = path.join(root, ".tools", "graphify", "Scripts", "graphify.exe");
const posixBin = path.join(root, ".tools", "graphify", "bin", "graphify");
const bin = [process.env.GRAPHIFY_BIN, winBin, posixBin].filter(Boolean).find((p) => fs.existsSync(p))
  || (spawnSync("graphify", ["--help"], { stdio: "ignore" }).status === 0 ? "graphify" : null);
const cli = bin === "graphify" ? "graphify"
  : bin ? path.relative(root, bin).replaceAll("\\", "/").replace(/\.exe$/, "") : null;

const head = () => spawnSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).stdout?.trim() || "";
const builtFrom = () => {
  try { return /Built from commit: `([0-9a-f]+)`/.exec(fs.readFileSync(report, "utf8"))?.[1] || ""; } catch { return ""; }
};
const fresh = () => { const b = builtFrom(); return !!b && head().startsWith(b); };

let note = "";
if (event === "SessionStart" && input.source !== "compact" && bin && !fresh()) {
  const r = spawnSync(bin, ["update", "."], { cwd: root, encoding: "utf8", timeout: 90000 });
  note = r.status === 0 ? "Graph rebuilt at session start (code-only, local)."
    : "Graph rebuild at session start failed; run `" + cli + " update .` and check the error.";
}

const section = (md, title) => {
  const m = new RegExp(`^## ${title}[^\\n]*\\n([\\s\\S]*?)(?=^## |(?![\\s\\S]))`, "m").exec(md);
  return m ? `## ${title}\n${m[1].trim()}\n` : "";
};

let ctx;
if (fs.existsSync(path.join(out, "graph.json")) && cli) {
  const md = fs.readFileSync(report, "utf8");
  const excerpt = ["Summary", "Graph Freshness", "God Nodes", "Surprising Connections"]
    .map((t) => section(md, t)).filter(Boolean).join("\n");
  ctx = [
    `graphify knowledge graph for this repo is at graphify-out/ (AGENTS.md "Codebase context (graphify)").`,
    note,
    fresh() ? "" : `WARNING: graph may be stale for HEAD; run \`${cli} update .\` before relying on it.`,
    `Start here: before any grep, glob, or source-file read for a codebase question, run ` +
      `\`${cli} query "<question>"\`, \`${cli} explain "<concept>"\`, or \`${cli} path "<A>" "<B>"\`. ` +
      `Read raw files only for lines the graph located. After changing code, run \`${cli} update .\`.`,
    event === "SessionStart"
      ? "When delegating, paste the relevant graph output (nodes, paths, file:line) into each assignment and tell the subagent to query the graph."
      : "You are a subagent: use the graph output in your assignment first, then query the graph yourself before exploring files.",
    "",
    "GRAPH_REPORT.md excerpt:",
    excerpt,
  ].filter((l) => l !== "").join("\n");
} else {
  ctx = `graphify is not set up in this checkout${cli ? " (no graph yet)" : " (CLI not found)"}. ` +
    `Before codebase exploration, set it up per AGENTS.md "Codebase context (graphify)": ` +
    "`python -m venv .tools/graphify`, `.tools/graphify/Scripts/pip install graphifyy==0.9.95` " +
    "(bin/pip on macOS/Linux), then `.tools/graphify/Scripts/graphify update .`. " +
    "Installing software needs user approval; if setup is not possible, say so and fall back to normal search.";
}

if (ctx.length > MAX) ctx = ctx.slice(0, MAX) + "\n[truncated; see graphify-out/GRAPH_REPORT.md]";
process.stdout.write(JSON.stringify({ hookSpecificOutput: { hookEventName: event, additionalContext: ctx } }));
process.exit(0);
