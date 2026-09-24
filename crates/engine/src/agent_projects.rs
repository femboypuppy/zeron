//! Project folders other coding agents have worked in on this device — the
//! New project flow's "Import from" source (`ListAgentProjects`).
//!
//! Every store is opened read-only and parsed best-effort: a missing, locked,
//! or unrecognized store contributes nothing rather than failing the listing.
//! Stores read, per agent (each honoring the agent's own directory override):
//!
//! - Claude Code — `.claude.json` `projects` keys and the `cwd` recorded in
//!   `projects/*/*.jsonl` transcripts (`$CLAUDE_CONFIG_DIR` or `~/.claude`).
//! - Codex — the `session_meta` `cwd` of `sessions/**/rollout-*.jsonl` and the
//!   `threads` table of `state_*.sqlite` (`$CODEX_HOME` or `~/.codex`).
//! - opencode — `session.directory` / `project.worktree` in `opencode.db`,
//!   plus the pre-SQLite JSON `storage/` (`$XDG_DATA_HOME/opencode`).
//! - pi — the session header `cwd` of `sessions/*/*.jsonl`
//!   (`$PI_CODING_AGENT_DIR` or `~/.pi/agent`).
//! - Cursor, Antigravity — the editors' `workspaceStorage/*/workspace.json`
//!   folder URIs.
//!
//! Only folders that still exist are reported; agent-managed scratch space
//! (temp and app-data dirs, `.<agent>/worktrees/…`) is skipped.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use rusqlite::OpenFlags;
use zeron_proto::{AgentProject, AgentProjectSource, HarnessId};

use crate::EngineError;
use crate::repos::{disposable_worker, home_dir};

/// Wall-clock ceiling for one listing — a wedged network home must fail the
/// listing, not the runtime (same disposable-worker shape as `ListDrives`).
const LIST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_PROJECTS_PER_SOURCE: usize = 500;
/// Newest-first cap on transcript files opened per store.
const MAX_SESSION_FILES: usize = 5_000;
/// Hard cap on directory entries collected while walking a session tree.
const MAX_WALK_ENTRIES: usize = 50_000;
/// Transcript lines read looking for a `cwd` before giving up on a file.
const MAX_HEADER_LINES: usize = 64;
/// Codex's first line embeds the base instructions, so lines can be large.
const MAX_LINE_BYTES: u64 = 8 * 1024 * 1024;
/// Transcript files tried per Claude/pi project directory.
const FILES_PER_PROJECT_DIR: usize = 5;

/// Where each agent keeps its history on this device.
pub struct AgentStoreRoots {
    pub claude_config: PathBuf,
    pub claude_json: PathBuf,
    pub codex_home: PathBuf,
    pub opencode_data: PathBuf,
    pub pi_agent: PathBuf,
    pub cursor_user: PathBuf,
    pub antigravity_user: PathBuf,
    /// Folders under these roots are never offered (temp and app-data dirs).
    pub excluded: Vec<PathBuf>,
}

impl AgentStoreRoots {
    /// Production resolution, mirroring each agent's own lookup.
    pub fn from_env() -> Self {
        let home = home_dir();
        let claude_dir = env_dir("CLAUDE_CONFIG_DIR");
        Self {
            claude_json: match &claude_dir {
                Some(dir) => dir.join(".claude.json"),
                None => home.join(".claude.json"),
            },
            claude_config: claude_dir.unwrap_or_else(|| home.join(".claude")),
            codex_home: env_dir("CODEX_HOME").unwrap_or_else(|| home.join(".codex")),
            opencode_data: env_dir("XDG_DATA_HOME")
                .unwrap_or_else(|| home.join(".local").join("share"))
                .join("opencode"),
            pi_agent: env_dir("PI_CODING_AGENT_DIR")
                .unwrap_or_else(|| home.join(".pi").join("agent")),
            cursor_user: editor_user_dir(&home, "Cursor"),
            antigravity_user: editor_user_dir(&home, "Antigravity"),
            excluded: excluded_roots(&home),
        }
    }
}

/// `ListAgentProjects`: every agent with at least one existing project folder.
pub async fn list_agent_projects() -> Result<Vec<AgentProjectSource>, EngineError> {
    let worker = disposable_worker("agent-projects", || discover(&AgentStoreRoots::from_env()));
    match tokio::time::timeout(LIST_TIMEOUT, worker).await {
        Ok(Some(sources)) => Ok(sources),
        Ok(None) => Err(EngineError::Other(
            "agent project listing worker exited".into(),
        )),
        Err(_) => Err(EngineError::Other(
            "agent project listing timed out on the device".into(),
        )),
    }
}

/// The blocking discovery pass over every agent store.
pub fn discover(roots: &AgentStoreRoots) -> Vec<AgentProjectSource> {
    [
        (HarnessId::ClaudeCode, claude_projects(roots)),
        (HarnessId::Codex, codex_projects(roots)),
        (HarnessId::Opencode, opencode_projects(roots)),
        (HarnessId::Cursor, editor_projects(&roots.cursor_user)),
        (HarnessId::Pi, pi_projects(roots)),
        (
            HarnessId::Antigravity,
            editor_projects(&roots.antigravity_user),
        ),
    ]
    .into_iter()
    .filter_map(|(harness, found)| {
        let projects = found.finish(&roots.excluded);
        (!projects.is_empty()).then_some(AgentProjectSource { harness, projects })
    })
    .collect()
}

// ── Per-agent readers ───────────────────────────────────────────────────────

fn claude_projects(roots: &AgentStoreRoots) -> Found {
    let mut found = Found::default();
    if let Some(config) = read_json(&roots.claude_json)
        && let Some(projects) = config.get("projects").and_then(|p| p.as_object())
    {
        for path in projects.keys() {
            found.add(path, None);
        }
    }
    // The transcript folder names are a lossy encoding of the path, so the
    // real path comes from the `cwd` the transcript lines record.
    for dir in subdirs(&roots.claude_config.join("projects")) {
        newest_transcript_cwd(&dir, &mut found, |line| str_field(line, "cwd"));
    }
    found
}

fn codex_projects(roots: &AgentStoreRoots) -> Found {
    let mut found = Found::default();
    let mut files = Vec::new();
    collect_files(&roots.codex_home.join("sessions"), 4, "jsonl", &mut files);
    collect_files(
        &roots.codex_home.join("archived_sessions"),
        1,
        "jsonl",
        &mut files,
    );
    newest_first(&mut files);
    files.truncate(MAX_SESSION_FILES);
    for (file, used) in files {
        let cwd = scan_jsonl(&file, 2, |line| {
            if line.get("type")?.as_str()? != "session_meta" {
                return None;
            }
            str_field(line.get("payload")?, "cwd")
        });
        if let Some(cwd) = cwd {
            found.add(&cwd, used);
        }
    }
    if let Some(db) = newest_codex_state_db(&roots.codex_home) {
        sqlite_paths(
            &db,
            "SELECT cwd, MAX(updated_at) FROM threads GROUP BY cwd",
            |secs| DateTime::from_timestamp(secs, 0),
            &mut found,
        );
    }
    found
}

fn opencode_projects(roots: &AgentStoreRoots) -> Found {
    let mut found = Found::default();
    let db = roots.opencode_data.join("opencode.db");
    if db.is_file() {
        sqlite_paths(
            &db,
            "SELECT directory, MAX(time_updated) FROM session GROUP BY directory",
            DateTime::from_timestamp_millis,
            &mut found,
        );
        sqlite_paths(
            &db,
            "SELECT worktree, time_updated FROM project",
            DateTime::from_timestamp_millis,
            &mut found,
        );
    }
    // Pre-SQLite releases: one JSON file per project and per session.
    let storage = roots.opencode_data.join("storage");
    let mut files = Vec::new();
    collect_files(&storage.join("project"), 1, "json", &mut files);
    collect_files(&storage.join("session"), 2, "json", &mut files);
    newest_first(&mut files);
    files.truncate(MAX_SESSION_FILES);
    for (file, modified) in files {
        let Some(record) = read_json(&file) else {
            continue;
        };
        let Some(path) = str_field(&record, "worktree").or_else(|| str_field(&record, "directory"))
        else {
            continue;
        };
        let used = record
            .get("time")
            .and_then(|t| t.get("updated"))
            .and_then(|t| t.as_i64())
            .and_then(DateTime::from_timestamp_millis)
            .or(modified);
        found.add(&path, used);
    }
    found
}

fn pi_projects(roots: &AgentStoreRoots) -> Found {
    let mut found = Found::default();
    for dir in subdirs(&roots.pi_agent.join("sessions")) {
        newest_transcript_cwd(&dir, &mut found, |line| str_field(line, "cwd"));
    }
    found
}

/// VS Code-family editors record each opened folder as a `file://` URI.
fn editor_projects(user_dir: &Path) -> Found {
    let mut found = Found::default();
    for dir in subdirs(&user_dir.join("workspaceStorage")) {
        let Some(workspace) = read_json(&dir.join("workspace.json")) else {
            continue;
        };
        let Some(path) = workspace
            .get("folder")
            .and_then(|f| f.as_str())
            .and_then(file_uri_path)
        else {
            continue;
        };
        let used = modified(&dir.join("state.vscdb")).or_else(|| modified(&dir));
        found.add(&path, used);
    }
    found
}

// ── Collection and normalization ────────────────────────────────────────────

/// Discovered folders keyed by normalized identity, keeping the latest use.
#[derive(Default)]
struct Found {
    entries: HashMap<String, (String, Option<DateTime<Utc>>)>,
}

impl Found {
    fn add(&mut self, raw: &str, used: Option<DateTime<Utc>>) {
        let Some(path) = normalize_path(raw) else {
            return;
        };
        let slot = self.entries.entry(path_key(&path)).or_insert((path, None));
        if used > slot.1 {
            slot.1 = used;
        }
    }

    /// Existing, non-excluded folders, most recently used first.
    fn finish(self, excluded: &[PathBuf]) -> Vec<AgentProject> {
        let mut projects: Vec<AgentProject> = self
            .entries
            .into_values()
            .filter_map(|(path, last_used_at)| {
                let folder = Path::new(&path);
                // A filesystem root is never a project.
                folder.parent()?;
                if excluded.iter().any(|root| path_under(&path, root))
                    || is_agent_worktree(folder)
                    || !folder.is_dir()
                {
                    return None;
                }
                Some(AgentProject {
                    name: folder.file_name()?.to_string_lossy().into_owned(),
                    is_repo: folder.join(".git").exists(),
                    path,
                    last_used_at,
                })
            })
            .collect();
        projects.sort_by(|a, b| {
            b.last_used_at
                .cmp(&a.last_used_at)
                .then_with(|| a.path.cmp(&b.path))
        });
        projects.truncate(MAX_PROJECTS_PER_SOURCE);
        projects
    }
}

/// The device-native absolute form of a recorded path, or `None` when it isn't
/// an absolute path here (e.g. a POSIX path recorded on another OS). Strips
/// Windows verbatim prefixes, unifies separators, uppercases drive letters,
/// and drops trailing separators.
pub(crate) fn normalize_path(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let raw = match raw.strip_prefix(r"\\?\UNC\") {
        Some(share) => format!(r"\\{share}"),
        None => raw.strip_prefix(r"\\?\").unwrap_or(raw).to_string(),
    };
    #[cfg(windows)]
    let raw = {
        let mut path = raw.replace('/', "\\");
        let bytes = path.as_bytes();
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            path[..1].make_ascii_uppercase();
        }
        path
    };
    let path = Path::new(&raw);
    if !path.is_absolute() {
        return None;
    }
    let clean: PathBuf = path.components().collect();
    Some(clean.to_string_lossy().into_owned())
}

/// Identity for dedupe: Windows paths compare case-insensitively.
pub(crate) fn path_key(path: &str) -> String {
    if cfg!(windows) {
        path.to_lowercase()
    } else {
        path.to_string()
    }
}

/// Segment-aware "is `path` at or under `root`", case-insensitive on Windows.
fn path_under(path: &str, root: &Path) -> bool {
    let Some(root) = normalize_path(&root.to_string_lossy()) else {
        return false;
    };
    let (path, root) = (path_key(path), path_key(&root));
    let root = root.trim_end_matches(std::path::MAIN_SEPARATOR);
    path == root
        || path
            .strip_prefix(root)
            .is_some_and(|rest| rest.starts_with(std::path::MAIN_SEPARATOR))
}

/// Agent-managed checkouts such as `~/.codex/worktrees/x` or
/// `repo/.claude/worktrees/y`: a hidden folder directly holding `worktrees`.
fn is_agent_worktree(path: &Path) -> bool {
    let names: Vec<_> = path
        .components()
        .filter_map(|c| match c {
            std::path::Component::Normal(name) => Some(name.to_string_lossy()),
            _ => None,
        })
        .collect();
    names
        .windows(2)
        .any(|pair| pair[0].starts_with('.') && pair[1].eq_ignore_ascii_case("worktrees"))
}

/// `file:///c%3A/Users/me` → `c:/Users/me`; `file:///home/me` → `/home/me`;
/// `file://host/share` → `//host/share`. Non-file URIs (remote workspaces)
/// yield `None`.
fn file_uri_path(uri: &str) -> Option<String> {
    let rest = uri.strip_prefix("file://")?;
    let decoded = percent_decode(rest)?;
    let bytes = decoded.as_bytes();
    if bytes.len() >= 3 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() && bytes[2] == b':' {
        return Some(decoded[1..].to_string());
    }
    if decoded.starts_with('/') {
        Some(decoded)
    } else {
        Some(format!("//{decoded}"))
    }
}

fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

// ── Store access helpers ────────────────────────────────────────────────────

fn env_dir(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The VS Code-family per-user settings dir for `app` on this platform.
fn editor_user_dir(home: &Path, app: &str) -> PathBuf {
    let base = if cfg!(windows) {
        env_dir("APPDATA").unwrap_or_else(|| home.join("AppData").join("Roaming"))
    } else if cfg!(target_os = "macos") {
        home.join("Library").join("Application Support")
    } else {
        env_dir("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config"))
    };
    base.join(app).join("User")
}

/// Temp and application-data roots: agents run scratch sessions there, but
/// nobody means them as projects.
fn excluded_roots(home: &Path) -> Vec<PathBuf> {
    let mut roots = vec![std::env::temp_dir()];
    if cfg!(windows) {
        roots.extend(env_dir("APPDATA"));
        roots.extend(env_dir("LOCALAPPDATA"));
    } else if cfg!(target_os = "macos") {
        roots.push(home.join("Library"));
    }
    roots
}

pub(crate) fn str_field(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_string)
}

pub(crate) fn read_json(path: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn modified(path: &Path) -> Option<DateTime<Utc>> {
    let time = std::fs::metadata(path).ok()?.modified().ok()?;
    Some(DateTime::<Utc>::from(time))
}

pub(crate) fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    read.flatten()
        .filter(|entry| entry.file_type().is_ok_and(|t| t.is_dir()))
        .map(|entry| entry.path())
        .collect()
}

/// Files with extension `ext` up to `depth` levels below `dir`, with mtimes.
pub(crate) fn collect_files(
    dir: &Path,
    depth: usize,
    ext: &str,
    out: &mut Vec<(PathBuf, Option<DateTime<Utc>>)>,
) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in read.flatten() {
        if out.len() >= MAX_WALK_ENTRIES {
            return;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            if depth > 1 {
                collect_files(&path, depth - 1, ext, out);
            }
        } else if kind.is_file() && path.extension().is_some_and(|e| e == ext) {
            let used = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .map(DateTime::<Utc>::from);
            out.push((path, used));
        }
    }
}

pub(crate) fn newest_first(files: &mut [(PathBuf, Option<DateTime<Utc>>)]) {
    files.sort_by(|a, b| b.1.cmp(&a.1));
}

/// A per-project transcript folder (Claude, pi): the folder's path comes from
/// the newest transcript that records one, dated by the newest transcript.
fn newest_transcript_cwd(
    dir: &Path,
    found: &mut Found,
    pick: impl Fn(&serde_json::Value) -> Option<String>,
) {
    let mut files = Vec::new();
    collect_files(dir, 1, "jsonl", &mut files);
    newest_first(&mut files);
    let used = files.first().and_then(|(_, used)| *used);
    let cwd = files
        .iter()
        .take(FILES_PER_PROJECT_DIR)
        .find_map(|(file, _)| scan_jsonl(file, MAX_HEADER_LINES, &pick));
    if let Some(cwd) = cwd {
        found.add(&cwd, used);
    }
}

/// The first `pick` hit among a JSONL file's first `max_lines` lines. An
/// oversized line ends the scan (the rest of the file can't be framed).
pub(crate) fn scan_jsonl(
    path: &Path,
    max_lines: usize,
    pick: impl Fn(&serde_json::Value) -> Option<String>,
) -> Option<String> {
    let mut reader = BufReader::new(std::fs::File::open(path).ok()?);
    let mut line = Vec::new();
    for _ in 0..max_lines {
        line.clear();
        let read = (&mut reader)
            .take(MAX_LINE_BYTES)
            .read_until(b'\n', &mut line)
            .ok()?;
        if read == 0 || (read as u64 == MAX_LINE_BYTES && !line.ends_with(b"\n")) {
            return None;
        }
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&line)
            && let Some(hit) = pick(&value)
        {
            return Some(hit);
        }
    }
    None
}

/// Codex versions its state database by file name (`state_5.sqlite`); the
/// highest version is the live one.
pub(crate) fn newest_codex_state_db(codex_home: &Path) -> Option<PathBuf> {
    std::fs::read_dir(codex_home)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let version: u32 = name
                .strip_prefix("state_")?
                .strip_suffix(".sqlite")?
                .parse()
                .ok()?;
            Some((version, entry.path()))
        })
        .max_by_key(|(version, _)| *version)
        .map(|(_, path)| path)
}

/// Rows of `(path, timestamp)` from a read-only SQLite query. A schema this
/// build doesn't recognize just yields nothing.
fn sqlite_paths(
    db: &Path,
    sql: &str,
    to_time: impl Fn(i64) -> Option<DateTime<Utc>>,
    found: &mut Found,
) {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let Ok(conn) = rusqlite::Connection::open_with_flags(db, flags) else {
        return;
    };
    let _ = conn.busy_timeout(Duration::from_millis(500));
    let Ok(mut stmt) = conn.prepare(sql) else {
        return;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        let path = row.get::<_, Option<String>>(0).ok().flatten();
        let time = row.get::<_, Option<i64>>(1).ok().flatten();
        Ok((path, time))
    }) else {
        return;
    };
    for (path, time) in rows.flatten() {
        if let Some(path) = path {
            found.add(&path, time.and_then(&to_time));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        roots: AgentStoreRoots,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().to_path_buf();
            let roots = AgentStoreRoots {
                claude_config: root.join("claude"),
                claude_json: root.join("claude.json"),
                codex_home: root.join("codex"),
                opencode_data: root.join("opencode"),
                pi_agent: root.join("pi"),
                cursor_user: root.join("cursor"),
                antigravity_user: root.join("antigravity"),
                excluded: Vec::new(),
            };
            Self {
                _dir: dir,
                root,
                roots,
            }
        }

        /// An existing project folder, returned in normalized form.
        fn project(&self, name: &str) -> String {
            let path = self.root.join("work").join(name);
            std::fs::create_dir_all(&path).unwrap();
            normalize_path(&path.to_string_lossy()).unwrap()
        }

        fn write(&self, rel: &str, contents: &str) {
            let path = self.root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }

        fn source(&self, harness: HarnessId) -> Vec<AgentProject> {
            discover(&self.roots)
                .into_iter()
                .find(|s| s.harness == harness)
                .map(|s| s.projects)
                .unwrap_or_default()
        }
    }

    fn paths(projects: &[AgentProject]) -> Vec<&str> {
        projects.iter().map(|p| p.path.as_str()).collect()
    }

    fn jsonl(lines: &[serde_json::Value]) -> String {
        lines.iter().map(|l| format!("{l}\n")).collect()
    }

    #[test]
    fn claude_reads_config_keys_and_transcript_cwds() {
        let fx = Fixture::new();
        let from_config = fx.project("alpha");
        let from_transcript = fx.project("beta");
        let gone = fx.root.join("work").join("deleted");
        fx.write(
            "claude.json",
            &json!({ "projects": {
                from_config.clone(): {},
                gone.to_string_lossy(): {},
            }})
            .to_string(),
        );
        fx.write(
            "claude/projects/encoded-beta/s1.jsonl",
            &jsonl(&[
                json!({"type": "queue-operation"}),
                json!({"type": "user", "cwd": from_transcript}),
            ]),
        );
        let projects = fx.source(HarnessId::ClaudeCode);
        let mut found = paths(&projects);
        found.sort();
        let mut expected = vec![from_config.as_str(), from_transcript.as_str()];
        expected.sort();
        assert_eq!(found, expected, "deleted folders are not offered");
        let beta = projects.iter().find(|p| p.path == from_transcript).unwrap();
        assert_eq!(beta.name, "beta");
        assert!(beta.last_used_at.is_some(), "dated by the transcript");
    }

    #[test]
    fn codex_reads_session_meta_and_state_db() {
        let fx = Fixture::new();
        let rollout = fx.project("from-rollout");
        let threaded = fx.project("from-db");
        fx.write(
            "codex/sessions/2026/06/19/rollout-1.jsonl",
            &jsonl(&[json!({"type": "session_meta", "payload": {"cwd": rollout}})]),
        );
        std::fs::create_dir_all(&fx.roots.codex_home).unwrap();
        let conn = rusqlite::Connection::open(fx.roots.codex_home.join("state_5.sqlite")).unwrap();
        conn.execute_batch("CREATE TABLE threads (cwd TEXT, updated_at INTEGER);")
            .unwrap();
        conn.execute(
            "INSERT INTO threads VALUES (?1, 1781842357)",
            [format!(r"\\?\{threaded}")],
        )
        .unwrap();
        drop(conn);
        // An older, stale state db must be ignored in favor of the newest.
        fx.write("codex/state_1.sqlite", "not a database");
        let projects = fx.source(HarnessId::Codex);
        assert_eq!(paths(&projects), vec![rollout.as_str(), threaded.as_str()]);
        assert_eq!(
            projects[1].last_used_at,
            DateTime::from_timestamp(1781842357, 0),
            "verbatim prefix stripped, db timestamp kept"
        );
    }

    #[test]
    fn opencode_reads_sqlite_sessions_and_legacy_storage() {
        let fx = Fixture::new();
        let session_dir = fx.project("oc-session");
        let legacy = fx.project("oc-legacy");
        std::fs::create_dir_all(&fx.roots.opencode_data).unwrap();
        let conn = rusqlite::Connection::open(fx.roots.opencode_data.join("opencode.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (directory TEXT, time_updated INTEGER);
             CREATE TABLE project (worktree TEXT, time_updated INTEGER);
             INSERT INTO project VALUES ('/', 1787847326320);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES (?1, 1787849430040)",
            [&session_dir],
        )
        .unwrap();
        drop(conn);
        fx.write(
            "opencode/storage/project/p1.json",
            &json!({"worktree": legacy, "time": {"updated": 1700000000000_i64}}).to_string(),
        );
        let projects = fx.source(HarnessId::Opencode);
        assert_eq!(
            paths(&projects),
            vec![session_dir.as_str(), legacy.as_str()],
            "the global `/` project is not a folder to import"
        );
    }

    #[test]
    fn pi_and_editors_are_discovered() {
        let fx = Fixture::new();
        let pi = fx.project("pi-proj");
        let cursor = fx.project("cursor proj");
        fx.write(
            "pi/sessions/--encoded--/s.jsonl",
            &jsonl(&[json!({"type": "session", "cwd": pi})]),
        );
        let uri = format!(
            "file://{}{}",
            if cursor.starts_with('/') { "" } else { "/" },
            cursor.replace('\\', "/").replace(' ', "%20")
        );
        fx.write(
            "cursor/workspaceStorage/abc/workspace.json",
            &json!({"folder": uri}).to_string(),
        );
        fx.write(
            "cursor/workspaceStorage/remote/workspace.json",
            &json!({"folder": "vscode-remote://ssh-remote+box/home/me"}).to_string(),
        );
        assert_eq!(paths(&fx.source(HarnessId::Pi)), vec![pi.as_str()]);
        assert_eq!(paths(&fx.source(HarnessId::Cursor)), vec![cursor.as_str()]);
        assert!(fx.source(HarnessId::Antigravity).is_empty());
    }

    #[test]
    fn duplicates_merge_to_latest_use_and_worktrees_are_skipped() {
        let fx = Fixture::new();
        let project = fx.project("dup");
        let worktree = fx.project(".codex/worktrees/wt");
        let mut found = Found::default();
        found.add(&project, DateTime::from_timestamp(10, 0));
        found.add(
            &format!("{project}{}", std::path::MAIN_SEPARATOR),
            DateTime::from_timestamp(20, 0),
        );
        found.add(&project, None);
        found.add(&worktree, DateTime::from_timestamp(30, 0));
        let projects = found.finish(&[]);
        assert_eq!(paths(&projects), vec![project.as_str()]);
        assert_eq!(projects[0].last_used_at, DateTime::from_timestamp(20, 0));

        let mut found = Found::default();
        found.add(&project, None);
        assert!(
            found.finish(&[fx.root.clone()]).is_empty(),
            "excluded roots"
        );
    }

    #[test]
    fn file_uris_decode_to_paths() {
        assert_eq!(
            file_uri_path("file:///c%3A/Users/me/my%20app").as_deref(),
            Some("c:/Users/me/my app")
        );
        assert_eq!(
            file_uri_path("file:///home/me/app").as_deref(),
            Some("/home/me/app")
        );
        assert_eq!(
            file_uri_path("file://server/share").as_deref(),
            Some("//server/share")
        );
        assert_eq!(file_uri_path("vscode-remote://ssh/x"), None);
        assert_eq!(file_uri_path("file:///bad%zz"), None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_paths_normalize() {
        assert_eq!(
            normalize_path(r"\\?\c:\Users\me\app\").as_deref(),
            Some(r"C:\Users\me\app")
        );
        assert_eq!(
            normalize_path("C:/Users/me/app").as_deref(),
            Some(r"C:\Users\me\app")
        );
        assert_eq!(normalize_path("/home/me"), None, "POSIX paths aren't local");
        assert_eq!(path_key(r"C:\Users\Me"), path_key(r"c:\users\me"));
    }

    #[cfg(unix)]
    #[test]
    fn unix_paths_normalize() {
        assert_eq!(
            normalize_path("/home/me/app/").as_deref(),
            Some("/home/me/app")
        );
        assert_eq!(normalize_path("relative/app"), None);
    }
}
