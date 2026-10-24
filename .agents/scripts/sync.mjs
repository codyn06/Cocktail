#!/usr/bin/env node
// Generates tool-specific agent config from canonical sources in .agents/.
// Usage: node .agents/scripts/sync.mjs [--check] [--force]
//   (default) validate, then write generated files and the manifest
//   --check   validate and report drift; write nothing; exit 1 on any problem
//   --force   overwrite/remove generated targets even if user-authored or edited
// Node >= 18, built-ins only.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = new Set(process.argv.slice(2));
const CHECK = args.has("--check");
const FORCE = args.has("--force");
for (const a of args) if (!["--check", "--force"].includes(a)) die(`unknown argument: ${a}`);

const MANIFEST = ".agents/generated-manifest.json";
const SCRIPT = ".agents/scripts/sync.mjs";
// Shell access enables inspection on hosts without native readers. Read-only shell
// behavior is instruction-enforced, not a security boundary.
const READONLY_CLAUDE_TOOLS = "Read, Grep, Glob, WebFetch, WebSearch, Bash";
const READONLY_CLAUDE_DISALLOWED = "Write, Edit, NotebookEdit";
const PI_READ_TOOLS = "read, grep, find, ls, bash";
const PI_EDIT_TOOLS = "read, grep, find, ls, bash, edit, write";

function die(msg) { console.error(`sync: ${msg}`); process.exit(2); }
const abs = (p) => path.join(ROOT, p);
const rel = (p) => path.relative(ROOT, p).split(path.sep).join("/");
const norm = (buf) => Buffer.from(buf.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
const sha = (buf) => crypto.createHash("sha256").update(norm(buf)).digest("hex");
const exists = (p) => fs.existsSync(abs(p));
const isDir = (p) => { try { return fs.statSync(abs(p)).isDirectory(); } catch { return false; } };
const inRoot = (p) => { const a = path.resolve(ROOT, p); return a.startsWith(ROOT + path.sep); };

function walk(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.isFile()) out.push(p);
  }
  return out.sort();
}

// Minimal YAML frontmatter parser: `key: value`, `key: [a, b]`, block scalars (> | >- |-),
// and quoted strings. Enough for SKILL.md and role files.
function parseFrontmatter(text) {
  text = text.replace(/\r\n/g, "\n");
  const m = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text);
  if (!m) return null;
  const data = {};
  const lines = m[1].split("\n");
  for (let i = 0; i < lines.length; i++) {
    const kv = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(lines[i]);
    if (!kv) continue;
    let [, key, val] = kv;
    if (/^[>|][-+]?$/.test(val)) {
      const block = [];
      while (i + 1 < lines.length && (/^\s+/.test(lines[i + 1]) || lines[i + 1] === "")) block.push(lines[++i].trim());
      val = val.startsWith(">") ? block.join(" ").trim() : block.join("\n").trim();
    } else if (/^\[.*\]$/.test(val)) {
      val = val.slice(1, -1).split(",").map((s) => unquote(s.trim())).filter(Boolean);
    } else val = unquote(val.trim());
    data[key] = val;
  }
  return { data, body: m[2] };
}
function unquote(s) {
  if (/^".*"$/.test(s)) { try { return JSON.parse(s); } catch { return s.slice(1, -1); } }
  if (/^'.*'$/.test(s)) return s.slice(1, -1).replace(/''/g, "'");
  return s;
}

// ---------- validation ----------
const errors = [];
const skillsDir = abs(".agents/skills");
const skills = {};
for (const e of fs.existsSync(skillsDir) ? fs.readdirSync(skillsDir, { withFileTypes: true }) : []) {
  if (!e.isDirectory()) continue;
  const f = `.agents/skills/${e.name}/SKILL.md`;
  if (!exists(f)) { errors.push(`${f}: missing`); continue; }
  const fm = parseFrontmatter(fs.readFileSync(abs(f), "utf8"));
  if (!fm) { errors.push(`${f}: no YAML frontmatter`); continue; }
  if (fm.data.name !== e.name) errors.push(`${f}: name "${fm.data.name}" != folder "${e.name}"`);
  if (!fm.data.description || typeof fm.data.description !== "string") errors.push(`${f}: description missing`);
  skills[e.name] = true;
}

const roles = [];
const rolesDir = abs(".agents/roles");
for (const file of fs.existsSync(rolesDir) ? fs.readdirSync(rolesDir).filter((f) => f.endsWith(".md")).sort() : []) {
  const f = `.agents/roles/${file}`;
  const fm = parseFrontmatter(fs.readFileSync(abs(f), "utf8"));
  if (!fm) { errors.push(`${f}: no YAML frontmatter`); continue; }
  const d = fm.data, base = file.slice(0, -3);
  if (d.name !== base) errors.push(`${f}: name "${d.name}" != file name "${base}"`);
  if (!d.description) errors.push(`${f}: description missing`);
  if (!["read-only", "edit"].includes(d.access)) errors.push(`${f}: access must be read-only|edit`);
  if (!Array.isArray(d.skills)) errors.push(`${f}: skills must be a [list]`);
  else for (const s of d.skills) if (!skills[s]) errors.push(`${f}: skill "${s}" not found in .agents/skills`);
  if (!fm.body.trim()) errors.push(`${f}: empty body`);
  if (fm.body.includes("'''")) errors.push(`${f}: body contains ''' (not allowed in TOML literal string)`);
  roles.push({ src: f, ...d, body: fm.body.trim() + "\n" });
}

let sources = null;
try { sources = JSON.parse(fs.readFileSync(abs(".agents/sources.json"), "utf8")); }
catch (e) { errors.push(`.agents/sources.json: ${e.message}`); }
if (sources) {
  for (const c of sources.components || []) {
    if (!c.destination || !exists(c.destination)) { errors.push(`sources.json ${c.name}: destination ${c.destination} missing`); continue; }
    const lic = c.licenseFiles?.length ? c.licenseFiles : ["LICENSE"];
    for (const l of lic) if (!exists(`${c.destination}/${l}`)) errors.push(`sources.json ${c.name}: ${c.destination}/${l} missing`);
  }
  for (const p of sources.projectSkills || [])
    if (!p.destination || !exists(p.destination)) errors.push(`sources.json projectSkills ${p.name}: destination missing`);
}
if (!exists("AGENTS.md")) errors.push("AGENTS.md missing");
for (const v of ["index.ts", "agents.ts"]) if (!exists(`.agents/vendor/pi-subagent/${v}`)) errors.push(`.agents/vendor/pi-subagent/${v} missing`);

if (errors.length) {
  console.error("Validation errors:\n" + errors.map((e) => "  - " + e).join("\n"));
  process.exit(1);
}

// ---------- desired outputs ----------
const out = new Map(); // relpath -> Buffer
const mdHeader = (src) => `<!-- GENERATED by ${SCRIPT} from ${src}; edit the source, then re-run sync -->\n`;
const yq = (s) => JSON.stringify(String(s)); // JSON string is a valid YAML double-quoted scalar
const skillNote = (r) => r.skills.length
  ? `Skills to load when relevant (read the SKILL.md before use): ${r.skills.map((s) => `\`.agents/skills/${s}/SKILL.md\``).join(", ")}.\n\n` : "";
const accessNote = (r) => r.access === "read-only"
  ? "Access: read-only. Do not create, edit, or delete files.\n\n" : "Access: edit (assigned files only).\n\n";

for (const name of Object.keys(skills).sort()) {
  const dir = abs(`.agents/skills/${name}`);
  for (const f of walk(dir)) {
    const r = rel(f).slice(`.agents/skills/${name}/`.length);
    const buf = fs.readFileSync(f);
    out.set(`.claude/skills/${name}/${r}`, buf);
    out.set(`.agent/skills/${name}/${r}`, buf);
  }
}

for (const r of roles) {
  const ro = r.access === "read-only";
  // Claude Code subagent
  let fm = `---\nname: ${r.name}\ndescription: ${yq(r.description)}\n`;
  if (ro) fm += `tools: ${READONLY_CLAUDE_TOOLS}\ndisallowedTools: ${READONLY_CLAUDE_DISALLOWED}\n`;
  fm += `model: inherit\n`;
  if (r.skills.length) fm += `skills:\n${r.skills.map((s) => `  - ${s}`).join("\n")}\n`;
  fm += `---\n`;
  out.set(`.claude/agents/${r.name}.md`, Buffer.from(fm + mdHeader(r.src) + "\n" + accessNote(r) + r.body));
  // Codex custom agent
  const toml = `# GENERATED by ${SCRIPT} from ${r.src}; edit the source, then re-run sync\n` +
    `name = ${JSON.stringify(r.name)}\ndescription = ${JSON.stringify(r.description)}\n` +
    `developer_instructions = '''\n${accessNote(r)}${skillNote(r)}${r.body}'''\n`;
  out.set(`.codex/agents/${r.name}.toml`, Buffer.from(toml));
  // Pi subagent
  const pi = `---\nname: ${r.name}\ndescription: ${yq(r.description)}\ntools: ${ro ? PI_READ_TOOLS : PI_EDIT_TOOLS}\n---\n` +
    mdHeader(r.src) + "\n" + accessNote(r) + skillNote(r) + r.body;
  out.set(`.pi/agents/${r.name}.md`, Buffer.from(pi));
}

out.set("GEMINI.md", Buffer.from(mdHeader("AGENTS.md") + "\n" + fs.readFileSync(abs("AGENTS.md"), "utf8")));
for (const v of ["index.ts", "agents.ts"]) {
  const src = `.agents/vendor/pi-subagent/${v}`;
  out.set(`.pi/extensions/subagent/${v}`,
    Buffer.from(`// GENERATED by ${SCRIPT} from ${src}; edit the source, then re-run sync\n` + fs.readFileSync(abs(src), "utf8")));
}

// ---------- plan ----------
let manifest = { files: {} };
if (exists(MANIFEST)) {
  try { manifest = JSON.parse(fs.readFileSync(abs(MANIFEST), "utf8")); manifest.files ||= {}; }
  catch (e) { die(`${MANIFEST} unreadable: ${e.message}`); }
}
const recorded = manifest.files;
const conflicts = [], writes = [], removals = [], notes = [];
const newFiles = {};

for (const [p, buf] of [...out].sort()) {
  const want = sha(buf);
  newFiles[p] = want;
  if (!exists(p)) { writes.push(p); notes.push(`missing: ${p}`); continue; }
  if (isDir(p)) { conflicts.push(`directory exists at generated file path: ${p}`); continue; }
  const cur = sha(fs.readFileSync(abs(p)));
  if (cur === want) { if (recorded[p] !== want) notes.push(`unrecorded (matches): ${p}`); continue; }
  if (!(p in recorded)) conflicts.push(`exists but not generated by sync (user-authored?): ${p}`);
  else if (cur !== recorded[p]) conflicts.push(`edited since generation: ${p}`);
  else notes.push(`out of date: ${p}`);
  writes.push(p);
}
for (const p of Object.keys(recorded).sort()) {
  if (p in newFiles) continue;
  if (!inRoot(p)) { conflicts.push(`manifest path outside repo, not removing: ${p}`); continue; }
  if (!exists(p)) { notes.push(`stale manifest entry: ${p}`); continue; }
  if (isDir(p)) { conflicts.push(`stale manifest path is a directory, not removing: ${p}`); continue; }
  if (sha(fs.readFileSync(abs(p))) !== recorded[p]) conflicts.push(`stale generated file was edited: ${p}`);
  else notes.push(`stale: ${p}`);
  removals.push(p);
}

const manifestText = JSON.stringify({
  comment: `GENERATED by ${SCRIPT}. Maps generated path -> sha256 of LF-normalized content.`,
  files: newFiles,
}, null, 2) + "\n";
const manifestOk = exists(MANIFEST) && sha(fs.readFileSync(abs(MANIFEST))) === sha(Buffer.from(manifestText));

if (CHECK) {
  const problems = [...conflicts.map((c) => "CONFLICT " + c), ...notes];
  if (!manifestOk && !problems.length) problems.push(`manifest out of date: ${MANIFEST}`);
  if (problems.length) {
    console.error(`sync --check: ${problems.length} problem(s):\n` + problems.map((p) => "  - " + p).join("\n"));
    process.exit(1);
  }
  console.log(`sync --check: OK (${out.size} generated files in sync, validation passed)`);
  process.exit(0);
}

if (conflicts.length && !FORCE) {
  console.error("sync: refusing to write; resolve or re-run with --force:\n" + conflicts.map((c) => "  - " + c).join("\n"));
  process.exit(1);
}
for (const p of writes) {
  fs.mkdirSync(path.dirname(abs(p)), { recursive: true });
  fs.writeFileSync(abs(p), out.get(p));
}
for (const p of removals) {
  if (!inRoot(p) || isDir(p)) { console.error(`sync: skipped unsafe removal: ${p}`); continue; }
  fs.rmSync(abs(p));
  // prune now-empty parent dirs up to the repo root
  let d = path.dirname(abs(p));
  while (d.startsWith(ROOT) && d !== ROOT && fs.readdirSync(d).length === 0) { fs.rmdirSync(d); d = path.dirname(d); }
}
fs.writeFileSync(abs(MANIFEST), manifestText);
console.log(`sync: wrote ${writes.length}, removed ${removals.length}, unchanged ${out.size - writes.length} (total ${out.size})`);
