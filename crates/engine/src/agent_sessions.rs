//! Conversations other coding agents recorded on this device: list them per
//! project folder, preview one, and import one into a Zeron chat.
//!
//! An import copies the conversation into a new chat's transcript, so it reads
//! like any Zeron session, and stores the agent's own session id as the chat's
//! harness session. The chat's next message therefore resumes the native
//! session and the agent keeps the full context. Supported stores (read-only):
//!
//! - Claude Code — `projects/<encoded cwd>/<session id>.jsonl`.
//! - Codex — `sessions/**/rollout-*-<thread id>.jsonl`, titled from the
//!   `threads` table / `session_index.jsonl`.
//! - opencode — the `session` / `message` / `part` tables of `opencode.db`.
//!
//! Tool calls map onto Zeron's typed [`ToolCall`]s (with the live drivers'
//! decoders where they exist) and pass the doc's usual privacy strip; tool
//! output is not copied, matching live transcripts.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use rusqlite::OpenFlags;
use serde_json::Value;
use zeron_doc::{
    MessagePart, MessageRole, MessageStatus, SessionDoc, SessionMessageEntry, continuation_id,
    sanitize_tool_call, split_parts,
};
use zeron_proto::{
    AgentPreviewMessage, AgentPreviewRole, AgentSession, AgentSessionPreview, Chat, ChatConfig,
    HarnessId, ImportedAgentSession, SandboxLevel, TodoItem, ToolCall,
};

use crate::EngineError;
use crate::agent_projects::{
    AgentStoreRoots, collect_files, newest_codex_state_db, newest_first, normalize_path, path_key,
    str_field, subdirs,
};
use crate::doc_host::DocHost;
use crate::repos::disposable_worker;
use crate::workspace_host::WorkspaceHost;

/// Wall-clock ceiling for a listing or preview (disposable worker, like
/// `ListAgentProjects`).
const READ_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_SESSIONS: usize = 200;
/// Newest-first cap on Codex rollouts opened while matching a folder.
const MAX_CODEX_ROLLOUTS: usize = 5_000;
/// Lines beyond this are skipped (a base64 image dump, not conversation).
const MAX_LINE_BYTES: usize = 64 * 1024 * 1024;
const PREVIEW_MESSAGES: usize = 150;
const PREVIEW_TEXT_MAX: usize = 2_000;
const PREVIEW_TOOLS_MAX: usize = 12;
const PREVIEW_TOOL_LABEL_MAX: usize = 120;
const TITLE_MAX: usize = 80;
/// Claude Code shortens encoded project dir names past this length.
const CLAUDE_DIR_NAME_MAX: usize = 200;

/// Whether conversations (not just project folders) can be imported from
/// `harness`.
pub fn supports_sessions(harness: HarnessId) -> bool {
    matches!(
        harness,
        HarnessId::ClaudeCode | HarnessId::Codex | HarnessId::Opencode
    )
}

/// `ListAgentSessions`: `harness`'s conversations recorded in `project`.
pub async fn list_sessions(
    harness: HarnessId,
    project: String,
) -> Result<Vec<AgentSession>, EngineError> {
    run_bounded("agent-sessions", move || {
        list_sessions_with(&AgentStoreRoots::from_env(), harness, &project)
    })
    .await
}

/// `PreviewAgentSession`: the conversation's latest messages.
pub async fn preview_session(
    harness: HarnessId,
    session_id: String,
) -> Result<AgentSessionPreview, EngineError> {
    run_bounded("agent-session-preview", move || {
        let transcript = load_transcript(&AgentStoreRoots::from_env(), harness, &session_id)?;
        Ok(preview(&transcript))
    })
    .await
}

/// `ImportAgentSession`: copy one conversation into a new chat that resumes it.
pub async fn import_session(
    doc_host: DocHost,
    workspace: WorkspaceHost,
    harness: HarnessId,
    session_id: String,
) -> Result<ImportedAgentSession, EngineError> {
    tokio::task::spawn_blocking(move || {
        import_session_with(
            &doc_host,
            &workspace,
            &AgentStoreRoots::from_env(),
            harness,
            &session_id,
        )
    })
    .await
    .map_err(|err| EngineError::Other(format!("conversation import failed: {err}")))?
}

async fn run_bounded<T: Send + 'static>(
    name: &'static str,
    work: impl FnOnce() -> Result<T, EngineError> + Send + 'static,
) -> Result<T, EngineError> {
    match tokio::time::timeout(READ_TIMEOUT, disposable_worker(name, work)).await {
        Ok(Some(result)) => result,
        Ok(None) => Err(EngineError::Other(format!("{name} worker exited"))),
        Err(_) => Err(EngineError::Other(
            "reading the agent's history timed out on the device".into(),
        )),
    }
}

pub fn list_sessions_with(
    roots: &AgentStoreRoots,
    harness: HarnessId,
    project: &str,
) -> Result<Vec<AgentSession>, EngineError> {
    let mut sessions = match harness {
        HarnessId::ClaudeCode => claude_sessions(roots, project),
        HarnessId::Codex => codex_sessions(roots, project),
        HarnessId::Opencode => opencode_sessions(roots, project),
        other => return Err(unsupported(other)),
    };
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    sessions.truncate(MAX_SESSIONS);
    Ok(sessions)
}

/// Import one conversation (blocking). Re-importing a conversation that
/// already has a chat on this device returns that chat instead.
pub fn import_session_with(
    doc_host: &DocHost,
    workspace: &WorkspaceHost,
    roots: &AgentStoreRoots,
    harness: HarnessId,
    session_id: &str,
) -> Result<ImportedAgentSession, EngineError> {
    let device = doc_host.device_id().to_string();
    let existing = workspace.read_chats()?.into_iter().find(|chat| {
        chat.device_id == device
            && chat.harness_session_id.as_deref() == Some(session_id)
            && chat.config.as_ref().is_some_and(|c| c.harness == harness)
    });
    if let Some(chat) = existing {
        return Ok(ImportedAgentSession {
            chat_id: chat.id,
            space_id: chat.space_id.unwrap_or_default(),
            existing: true,
        });
    }

    let transcript = load_transcript(roots, harness, session_id)?;
    let cwd = transcript
        .cwd
        .as_deref()
        .map(strip_verbatim)
        .filter(|cwd| !cwd.trim().is_empty())
        .ok_or_else(|| EngineError::Other("the conversation has no working folder".into()))?;
    let folder = normalize_path(&cwd)
        .filter(|folder| Path::new(folder).is_dir())
        .ok_or_else(|| {
            EngineError::Other(format!("the conversation's folder no longer exists: {cwd}"))
        })?;

    let space_id = match workspace
        .read_spaces()?
        .into_iter()
        .find(|s| s.device_id == device && same_folder(&s.path, &folder))
    {
        Some(space) => space.id,
        None => {
            let id = crate::new_id();
            let is_repo = Path::new(&folder).join(".git").exists();
            workspace.create_space(&id, &device, &folder, None, is_repo)?;
            id
        }
    };

    // Transcript first, row last (the local-import order): the row is what
    // makes the chat visible, and its first open must find the seeded doc.
    let chat_id = crate::new_id();
    let doc = SessionDoc::init(&chat_id)?;
    for entry in doc_entries(&transcript, &device) {
        doc.push_message(&entry)?;
    }
    doc_host.seed_chat_snapshot(&chat_id, &doc.export_snapshot()?)?;

    let created_at = transcript.started_at().unwrap_or_else(Utc::now);
    let last_at = transcript.ended_at().unwrap_or(created_at);
    let title = transcript
        .title
        .clone()
        .filter(|t| !t.trim().is_empty())
        .or_else(|| transcript.first_prompt().map(|p| one_line(p, TITLE_MAX)))
        .unwrap_or_else(|| "Imported conversation".into());
    workspace.import_chat_row(&Chat {
        id: chat_id.clone(),
        device_id: device,
        title: Some(title),
        archived: false,
        cwd: Some(cwd.clone()),
        branch: None,
        checkout_id: None,
        source_context: None,
        config: Some(ChatConfig {
            harness,
            model: None,
            reasoning: None,
            model_options: Default::default(),
            sandbox: SandboxLevel::WorkspaceWrite,
        }),
        last_message_preview: None,
        last_message_at: Some(last_at),
        created_at,
        // Resume is only injected when the run's cwd equals this, and the
        // run launches from the row's `cwd` — so both carry the same string.
        harness_session_id: Some(session_id.to_string()),
        harness_session_cwd: Some(cwd),
        space_id: Some(space_id.clone()),
        // Already read: an import must not raise a "finished, unseen" badge.
        last_seen_at: Some(last_at),
        room_gen: Some(2),
        parent_chat_id: None,
    })?;
    Ok(ImportedAgentSession {
        chat_id,
        space_id,
        existing: false,
    })
}

fn load_transcript(
    roots: &AgentStoreRoots,
    harness: HarnessId,
    session_id: &str,
) -> Result<Transcript, EngineError> {
    if !valid_session_id(session_id) {
        return Err(EngineError::Other(format!(
            "invalid session id: {session_id}"
        )));
    }
    match harness {
        HarnessId::ClaudeCode => claude_transcript(&claude_session_file(roots, session_id)?),
        HarnessId::Codex => codex_transcript(&codex_session_file(roots, session_id)?),
        HarnessId::Opencode => opencode_transcript(roots, session_id),
        other => Err(unsupported(other)),
    }
}

fn unsupported(harness: HarnessId) -> EngineError {
    EngineError::Other(format!(
        "importing conversations from {harness:?} is not supported"
    ))
}

/// Session ids name files and rows; anything path-shaped is refused.
fn valid_session_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn not_found(session_id: &str) -> EngineError {
    EngineError::Other(format!("conversation {session_id} was not found"))
}

// ── Normalized transcript ───────────────────────────────────────────────────

/// A conversation as prompt-led turns: each turn is one user prompt and
/// everything the agent did in reply.
#[derive(Default)]
struct Transcript {
    turns: Vec<Turn>,
    title: Option<String>,
    cwd: Option<String>,
}

#[derive(Default)]
struct Turn {
    prompt: Option<String>,
    started_at: Option<DateTime<Utc>>,
    ended_at: Option<DateTime<Utc>>,
    items: Vec<Item>,
}

enum Item {
    Text(String),
    Reasoning(String),
    Tool {
        id: String,
        call: ToolCall,
        is_error: bool,
    },
}

impl Transcript {
    fn prompt(&mut self, text: String, at: Option<DateTime<Utc>>) {
        self.turns.push(Turn {
            prompt: Some(text),
            started_at: at,
            ended_at: at,
            items: Vec::new(),
        });
    }

    /// The turn agent output lands in (a leading reply gets a prompt-less turn).
    fn current(&mut self, at: Option<DateTime<Utc>>) -> &mut Turn {
        if self.turns.is_empty() {
            self.turns.push(Turn {
                started_at: at,
                ..Turn::default()
            });
        }
        let turn = self.turns.last_mut().expect("a turn exists");
        if at > turn.ended_at {
            turn.ended_at = at;
        }
        turn
    }

    fn text(&mut self, text: &str, at: Option<DateTime<Utc>>) {
        if text.trim().is_empty() {
            return;
        }
        let turn = self.current(at);
        match turn.items.last_mut() {
            Some(Item::Text(prev)) => {
                prev.push_str("\n\n");
                prev.push_str(text);
            }
            _ => turn.items.push(Item::Text(text.to_string())),
        }
    }

    fn reasoning(&mut self, text: &str, at: Option<DateTime<Utc>>) {
        if text.trim().is_empty() {
            return;
        }
        let turn = self.current(at);
        match turn.items.last_mut() {
            Some(Item::Reasoning(prev)) => {
                prev.push_str("\n\n");
                prev.push_str(text);
            }
            _ => turn.items.push(Item::Reasoning(text.to_string())),
        }
    }

    fn tool(&mut self, id: String, call: ToolCall, at: Option<DateTime<Utc>>) {
        self.current(at).items.push(Item::Tool {
            id,
            call,
            is_error: false,
        });
    }

    fn tool_result(&mut self, id: &str, is_error: bool, at: Option<DateTime<Utc>>) {
        if self.turns.is_empty() {
            return;
        }
        self.current(at);
        for turn in self.turns.iter_mut().rev() {
            for item in turn.items.iter_mut().rev() {
                if let Item::Tool {
                    id: tool_id,
                    is_error: failed,
                    ..
                } = item
                    && tool_id == id
                {
                    *failed = is_error;
                    return;
                }
            }
        }
    }

    fn prompt_count(&self) -> usize {
        self.turns.iter().filter(|t| t.prompt.is_some()).count()
    }

    fn first_prompt(&self) -> Option<&str> {
        self.turns.iter().find_map(|t| t.prompt.as_deref())
    }

    fn started_at(&self) -> Option<DateTime<Utc>> {
        self.turns.iter().find_map(|t| t.started_at.or(t.ended_at))
    }

    fn ended_at(&self) -> Option<DateTime<Utc>> {
        self.turns
            .iter()
            .rev()
            .find_map(|t| t.ended_at.or(t.started_at))
    }
}

/// The transcript as chat doc entries: a user entry per prompt and one
/// completed assistant entry per reply (split into continuations when it
/// exceeds the inline cap). Timestamps never run backwards.
fn doc_entries(transcript: &Transcript, device_id: &str) -> Vec<SessionMessageEntry> {
    let mut entries = Vec::new();
    let mut clock = 0_i64;
    let mut stamp = |at: Option<DateTime<Utc>>| {
        clock = at.map_or(clock, |at| at.timestamp_millis()).max(clock);
        clock
    };
    for turn in &transcript.turns {
        if let Some(prompt) = &turn.prompt {
            entries.push(SessionMessageEntry {
                id: crate::new_id(),
                role: MessageRole::User,
                parts: vec![MessagePart::Text {
                    id: "t0".into(),
                    text: prompt.clone(),
                }],
                created_at: stamp(turn.started_at),
                device_id: device_id.to_string(),
                status: Some(MessageStatus::Complete),
                continuation_of: None,
                duration_ms: None,
            });
        }
        let parts = turn_parts(turn);
        if parts.is_empty() {
            continue;
        }
        let created_at = stamp(turn.started_at.or(turn.ended_at));
        stamp(turn.ended_at);
        let duration_ms = match (turn.started_at, turn.ended_at) {
            (Some(start), Some(end)) if end > start => Some((end - start).num_milliseconds()),
            _ => None,
        };
        let root = crate::new_id();
        for (index, chunk) in split_parts(&parts).into_iter().enumerate() {
            entries.push(SessionMessageEntry {
                id: if index == 0 {
                    root.clone()
                } else {
                    continuation_id(&root, index)
                },
                role: MessageRole::Assistant,
                parts: chunk,
                created_at,
                device_id: device_id.to_string(),
                status: Some(MessageStatus::Complete),
                continuation_of: (index > 0).then(|| root.clone()),
                duration_ms,
            });
        }
    }
    entries
}

fn turn_parts(turn: &Turn) -> Vec<MessagePart> {
    let mut parts = Vec::new();
    let mut seen = HashSet::new();
    let (mut texts, mut thoughts) = (0, 0);
    for (index, item) in turn.items.iter().enumerate() {
        match item {
            Item::Text(text) => {
                parts.push(MessagePart::Text {
                    id: format!("t{texts}"),
                    text: text.clone(),
                });
                texts += 1;
            }
            Item::Reasoning(text) => {
                parts.push(MessagePart::Reasoning {
                    id: format!("r{thoughts}"),
                    text: text.clone(),
                });
                thoughts += 1;
            }
            Item::Tool { id, call, is_error } => {
                let id = if id.is_empty() || !seen.insert(id.clone()) {
                    format!("tool-{index}")
                } else {
                    id.clone()
                };
                parts.push(MessagePart::Tool {
                    id,
                    call: sanitize_tool_call(call),
                    is_error: *is_error,
                    // History is settled: an unresolved chip would spin forever.
                    resolved: true,
                    output: None,
                    diff: None,
                    output_ref: None,
                    output_bytes: None,
                    diff_ref: None,
                    diff_stats: None,
                    subagent_ref: None,
                    subagent_status: None,
                    subagent_tail: None,
                });
            }
        }
    }
    parts
}

/// The latest [`PREVIEW_MESSAGES`] messages: prose clipped, tools as labels,
/// reasoning left out.
fn preview(transcript: &Transcript) -> AgentSessionPreview {
    let mut messages = Vec::new();
    for turn in &transcript.turns {
        if let Some(prompt) = &turn.prompt {
            messages.push(AgentPreviewMessage {
                role: AgentPreviewRole::User,
                text: clip(prompt, PREVIEW_TEXT_MAX),
                tools: Vec::new(),
                at: turn.started_at,
            });
        }
        let mut texts = Vec::new();
        let mut tools = Vec::new();
        for item in &turn.items {
            match item {
                Item::Text(text) => texts.push(text.as_str()),
                Item::Tool { call, .. } => tools.push(tool_label(call)),
                Item::Reasoning(_) => {}
            }
        }
        if texts.is_empty() && tools.is_empty() {
            continue;
        }
        if tools.len() > PREVIEW_TOOLS_MAX {
            let more = tools.len() - PREVIEW_TOOLS_MAX;
            tools.truncate(PREVIEW_TOOLS_MAX);
            tools.push(format!("+{more} more"));
        }
        messages.push(AgentPreviewMessage {
            role: AgentPreviewRole::Assistant,
            text: clip(&texts.join("\n\n"), PREVIEW_TEXT_MAX),
            tools,
            at: turn.ended_at,
        });
    }
    let omitted = messages.len().saturating_sub(PREVIEW_MESSAGES);
    messages.drain(..omitted);
    AgentSessionPreview { messages, omitted }
}

fn tool_label(call: &ToolCall) -> String {
    let (label, detail) = zeron_proto::view::tool_chip_content(call);
    if detail.is_empty() {
        label.to_string()
    } else {
        one_line(&format!("{label} {detail}"), PREVIEW_TOOL_LABEL_MAX)
    }
}

/// `text` cut to at most `max` chars (on a char boundary), marked with `…`.
fn clip(text: &str, max: usize) -> String {
    let text = text.trim();
    match text.char_indices().nth(max) {
        Some((cut, _)) => format!("{}…", text[..cut].trim_end()),
        None => text.to_string(),
    }
}

/// The first line of `text`, clipped.
fn one_line(text: &str, max: usize) -> String {
    clip(text.trim().lines().next().unwrap_or_default(), max)
}

fn same_folder(a: &str, b: &str) -> bool {
    match (normalize_path(a), normalize_path(b)) {
        (Some(a), Some(b)) => path_key(&a) == path_key(&b),
        _ => false,
    }
}

/// `\\?\C:\x` → `C:\x` (Codex records verbatim paths).
fn strip_verbatim(path: &str) -> String {
    match path.strip_prefix(r"\\?\UNC\") {
        Some(share) => format!(r"\\{share}"),
        None => path.strip_prefix(r"\\?\").unwrap_or(path).to_string(),
    }
}

fn timestamp(line: &Value) -> Option<DateTime<Utc>> {
    let raw = line.get("timestamp")?.as_str()?;
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

/// Stream a JSONL file, handing each parseable line to `each`. With
/// `needles`, only lines containing one of them are parsed (a cheap prefilter
/// for listings over large transcripts).
fn for_each_line(
    path: &Path,
    needles: &[&str],
    mut each: impl FnMut(Value),
) -> Result<(), EngineError> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        if line.len() > MAX_LINE_BYTES {
            continue;
        }
        if !needles.is_empty() {
            let Ok(text) = std::str::from_utf8(&line) else {
                continue;
            };
            if !needles.iter().any(|needle| text.contains(needle)) {
                continue;
            }
        }
        if let Ok(value) = serde_json::from_slice::<Value>(&line) {
            each(value);
        }
    }
}

fn first_line(path: &Path) -> Option<Value> {
    let mut reader = BufReader::new(std::fs::File::open(path).ok()?).take(MAX_LINE_BYTES as u64);
    let mut line = Vec::new();
    reader.read_until(b'\n', &mut line).ok()?;
    serde_json::from_slice(&line).ok()
}

fn file_contains(path: &Path, needle: &str) -> Result<bool, EngineError> {
    let mut found = false;
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut line = Vec::new();
    while !found {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        found = std::str::from_utf8(&line).is_ok_and(|text| text.contains(needle));
    }
    Ok(found)
}

fn file_stem(path: &Path) -> Option<String> {
    Some(path.file_stem()?.to_string_lossy().into_owned())
}

// ── Claude Code ─────────────────────────────────────────────────────────────

/// Claude Code's project dir name for a cwd: every non-alphanumeric char → `-`.
fn claude_dir_name(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

fn claude_project_dirs(roots: &AgentStoreRoots, project: &str) -> Vec<PathBuf> {
    let projects = roots.claude_config.join("projects");
    let wanted: Vec<String> = [Some(project.to_string()), normalize_path(project)]
        .into_iter()
        .flatten()
        .map(|p| claude_dir_name(&p))
        .collect();
    let matches = |name: &str| {
        wanted.iter().any(|want| {
            let (name, want) = if cfg!(windows) {
                (name.to_ascii_lowercase(), want.to_ascii_lowercase())
            } else {
                (name.to_string(), want.clone())
            };
            name == want
                || (want.len() > CLAUDE_DIR_NAME_MAX
                    && name.starts_with(&want[..CLAUDE_DIR_NAME_MAX]))
        })
    };
    subdirs(&projects)
        .into_iter()
        .filter(|dir| {
            dir.file_name()
                .is_some_and(|name| matches(&name.to_string_lossy()))
        })
        .collect()
}

fn claude_sessions(roots: &AgentStoreRoots, project: &str) -> Vec<AgentSession> {
    let mut files = Vec::new();
    for dir in claude_project_dirs(roots, project) {
        collect_files(&dir, 1, "jsonl", &mut files);
    }
    newest_first(&mut files);
    files.truncate(MAX_SESSIONS);
    files
        .into_iter()
        .filter_map(|(file, updated_at)| {
            let summary = claude_summary(&file).ok()?;
            let cwd = summary.cwd.unwrap_or_else(|| project.to_string());
            if summary.prompts == 0 || !same_folder(&cwd, project) {
                return None;
            }
            Some(AgentSession {
                id: file_stem(&file)?,
                title: summary
                    .title
                    .or_else(|| summary.first_prompt.map(|p| one_line(&p, TITLE_MAX)))
                    .unwrap_or_else(|| "Untitled conversation".into()),
                cwd,
                updated_at,
                prompt_count: summary.prompts,
            })
        })
        .collect()
}

#[derive(Default)]
struct Summary {
    title: Option<String>,
    first_prompt: Option<String>,
    prompts: usize,
    cwd: Option<String>,
}

/// Title, prompt count, and cwd without parsing the agent's output lines.
fn claude_summary(path: &Path) -> Result<Summary, EngineError> {
    let mut summary = Summary::default();
    let (mut custom, mut ai) = (None, None);
    for_each_line(
        path,
        &["\"type\":\"user\"", "\"custom-title\"", "\"ai-title\""],
        |line| match line.get("type").and_then(Value::as_str) {
            Some("user") if !claude_sidechain(&line) => {
                if summary.cwd.is_none() {
                    summary.cwd = str_field(&line, "cwd");
                }
                if let (Some(prompt), _) = claude_user_line(&line) {
                    summary.prompts += 1;
                    summary.first_prompt.get_or_insert(prompt);
                }
            }
            Some("custom-title") => custom = str_field(&line, "customTitle").or(custom.take()),
            Some("ai-title") => ai = str_field(&line, "aiTitle").or(ai.take()),
            _ => {}
        },
    )?;
    summary.title = custom.or(ai);
    Ok(summary)
}

fn claude_session_file(roots: &AgentStoreRoots, session_id: &str) -> Result<PathBuf, EngineError> {
    subdirs(&roots.claude_config.join("projects"))
        .into_iter()
        .map(|dir| dir.join(format!("{session_id}.jsonl")))
        .find(|file| file.is_file())
        .ok_or_else(|| not_found(session_id))
}

fn claude_sidechain(line: &Value) -> bool {
    line.get("isSidechain").and_then(Value::as_bool) == Some(true)
}

fn claude_transcript(path: &Path) -> Result<Transcript, EngineError> {
    let mut transcript = Transcript::default();
    let (mut custom, mut ai) = (None, None);
    for_each_line(path, &[], |line| {
        if claude_sidechain(&line) {
            return;
        }
        let at = timestamp(&line);
        match line.get("type").and_then(Value::as_str) {
            Some("user") => {
                if transcript.cwd.is_none() {
                    transcript.cwd = str_field(&line, "cwd");
                }
                let (prompt, results) = claude_user_line(&line);
                for (id, is_error) in results {
                    transcript.tool_result(&id, is_error, at);
                }
                if let Some(prompt) = prompt {
                    transcript.prompt(prompt, at);
                }
            }
            Some("assistant") => {
                if transcript.cwd.is_none() {
                    transcript.cwd = str_field(&line, "cwd");
                }
                let content = line.pointer("/message/content");
                if let Some(text) = content.and_then(Value::as_str) {
                    transcript.text(text, at);
                }
                for block in content.and_then(Value::as_array).into_iter().flatten() {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            transcript.text(block["text"].as_str().unwrap_or_default(), at)
                        }
                        Some("thinking") => {
                            transcript.reasoning(block["thinking"].as_str().unwrap_or_default(), at)
                        }
                        Some("tool_use") => {
                            let name = block["name"].as_str().unwrap_or_default();
                            transcript.tool(
                                block["id"].as_str().unwrap_or_default().to_string(),
                                zeron_harness::claude::decode_tool_use(name, &block["input"]),
                                at,
                            );
                        }
                        _ => {}
                    }
                }
            }
            Some("custom-title") => custom = str_field(&line, "customTitle").or(custom.take()),
            Some("ai-title") => ai = str_field(&line, "aiTitle").or(ai.take()),
            _ => {}
        }
    })?;
    transcript.title = custom.or(ai);
    Ok(transcript)
}

/// A Claude `user` line: the prompt the user typed (if any), plus the tool
/// results it carries as `(tool_use_id, is_error)`.
fn claude_user_line(line: &Value) -> (Option<String>, Vec<(String, bool)>) {
    let flagged = |key: &str| line.get(key).and_then(Value::as_bool) == Some(true);
    if flagged("isMeta") || flagged("isCompactSummary") || flagged("isVisibleInTranscriptOnly") {
        return (None, Vec::new());
    }
    let Some(content) = line.pointer("/message/content") else {
        return (None, Vec::new());
    };
    if let Some(text) = content.as_str() {
        return (claude_prompt_text(text), Vec::new());
    }
    let mut texts = Vec::new();
    let mut images = 0;
    let mut results = Vec::new();
    for block in content.as_array().into_iter().flatten() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => texts.extend(claude_prompt_text(
                block["text"].as_str().unwrap_or_default(),
            )),
            Some("image") => images += 1,
            Some("tool_result") => results.push((
                block["tool_use_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                block["is_error"].as_bool() == Some(true),
            )),
            _ => {}
        }
    }
    if images > 0 && results.is_empty() {
        texts.push(if images == 1 {
            "[image]".into()
        } else {
            format!("[{images} images]")
        });
    }
    let prompt = (!texts.is_empty()).then(|| texts.join("\n\n"));
    (prompt, results)
}

/// The conversational part of a user text block: context injections and
/// local-command noise are dropped, slash commands read as typed.
fn claude_prompt_text(raw: &str) -> Option<String> {
    let text = strip_tag_blocks(raw, "system-reminder");
    let text = text.trim();
    if text.is_empty() || zeron_harness::claude::is_synthetic_user_text(text) {
        return None;
    }
    const NOISE: [&str; 6] = [
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<bash-stdout>",
        "<bash-stderr>",
        "<task-notification>",
    ];
    if NOISE.iter().any(|tag| text.starts_with(tag)) {
        return None;
    }
    if let Some(name) = tag_body(text, "command-name") {
        let args = tag_body(text, "command-args").unwrap_or_default();
        return Some(format!("{name} {args}").trim().to_string());
    }
    if let Some(command) = tag_body(text, "bash-input") {
        return Some(format!("! {command}"));
    }
    Some(text.to_string())
}

fn tag_body<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = start + text[start..].find(&close)?;
    Some(text[start..end].trim())
}

fn strip_tag_blocks(text: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        out.push_str(&rest[..start]);
        match rest[start..].find(&close) {
            Some(end) => rest = &rest[start + end + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

// ── Codex ───────────────────────────────────────────────────────────────────

/// `(thread id, cwd)` from a rollout's `session_meta` header.
fn codex_meta(path: &Path) -> Option<(String, String)> {
    let line = first_line(path)?;
    if line.get("type")?.as_str()? != "session_meta" {
        return None;
    }
    let payload = line.get("payload")?;
    let id = str_field(payload, "id").or_else(|| str_field(payload, "session_id"))?;
    Some((id, str_field(payload, "cwd")?))
}

fn codex_rollouts(roots: &AgentStoreRoots) -> Vec<(PathBuf, Option<DateTime<Utc>>)> {
    let mut files = Vec::new();
    collect_files(&roots.codex_home.join("sessions"), 4, "jsonl", &mut files);
    newest_first(&mut files);
    files.truncate(MAX_CODEX_ROLLOUTS);
    files
}

/// Thread names: the state db's `threads` (a user rename wins over the
/// generated title), then `session_index.jsonl`.
fn codex_titles(roots: &AgentStoreRoots) -> HashMap<String, String> {
    let mut titles = HashMap::new();
    if let Some(db) = newest_codex_state_db(&roots.codex_home)
        && let Some(conn) = open_readonly(&db)
    {
        for sql in [
            "SELECT id, COALESCE(NULLIF(name, ''), title) FROM threads",
            "SELECT id, title FROM threads",
        ] {
            if query_pairs(&conn, sql, &mut titles) {
                break;
            }
        }
    }
    let index = roots.codex_home.join("session_index.jsonl");
    if index.is_file() {
        let _ = for_each_line(&index, &[], |line| {
            if let (Some(id), Some(name)) =
                (str_field(&line, "id"), str_field(&line, "thread_name"))
                && !name.trim().is_empty()
            {
                titles.entry(id).or_insert(name);
            }
        });
    }
    titles.retain(|_, title| !title.trim().is_empty());
    titles
}

fn codex_sessions(roots: &AgentStoreRoots, project: &str) -> Vec<AgentSession> {
    let titles = codex_titles(roots);
    let mut sessions = Vec::new();
    for (file, updated_at) in codex_rollouts(roots) {
        if sessions.len() >= MAX_SESSIONS {
            break;
        }
        let Some((id, cwd)) = codex_meta(&file) else {
            continue;
        };
        if !same_folder(&strip_verbatim(&cwd), project) {
            continue;
        }
        let mut prompts = 0;
        let mut first_prompt = None;
        let _ = for_each_line(&file, &["\"user_message\""], |line| {
            if line.pointer("/payload/type").and_then(Value::as_str) == Some("user_message")
                && let Some(message) = line.pointer("/payload/message").and_then(Value::as_str)
                && !message.trim().is_empty()
            {
                prompts += 1;
                first_prompt.get_or_insert_with(|| message.to_string());
            }
        });
        if prompts == 0 {
            // Older rollouts carry prompts only as response items.
            let Ok(transcript) = codex_transcript(&file) else {
                continue;
            };
            prompts = transcript.prompt_count();
            first_prompt = transcript.first_prompt().map(str::to_string);
        }
        if prompts == 0 {
            continue;
        }
        sessions.push(AgentSession {
            title: titles
                .get(&id)
                .cloned()
                .or_else(|| first_prompt.map(|p| one_line(&p, TITLE_MAX)))
                .unwrap_or_else(|| "Untitled conversation".into()),
            id,
            cwd: strip_verbatim(&cwd),
            updated_at,
            prompt_count: prompts,
        });
    }
    sessions
}

fn codex_session_file(roots: &AgentStoreRoots, session_id: &str) -> Result<PathBuf, EngineError> {
    let rollouts = codex_rollouts(roots);
    let suffix = format!("{session_id}.jsonl");
    rollouts
        .iter()
        .map(|(file, _)| file)
        .find(|file| {
            file.file_name()
                .is_some_and(|name| name.to_string_lossy().ends_with(&suffix))
        })
        .or_else(|| {
            rollouts
                .iter()
                .map(|(file, _)| file)
                .find(|file| codex_meta(file).is_some_and(|(id, _)| id == session_id))
        })
        .cloned()
        .ok_or_else(|| not_found(session_id))
}

fn codex_transcript(path: &Path) -> Result<Transcript, EngineError> {
    // Prompts come from `user_message` events: the raw `user` response items
    // also carry injected context (environment, AGENTS.md). Rollouts written
    // before those events existed fall back to the response items.
    let events = file_contains(path, "\"user_message\"")?;
    let mut transcript = Transcript::default();
    for_each_line(path, &[], |line| {
        let at = timestamp(&line);
        let Some(payload) = line.get("payload") else {
            return;
        };
        let kind = payload.get("type").and_then(Value::as_str).unwrap_or("");
        match (line.get("type").and_then(Value::as_str), kind) {
            (Some("session_meta"), _) => {
                transcript.cwd = str_field(payload, "cwd").or(transcript.cwd.take());
            }
            (Some("event_msg"), "user_message") if events => {
                if let Some(message) = payload.get("message").and_then(Value::as_str)
                    && !message.trim().is_empty()
                {
                    transcript.prompt(message.trim().to_string(), at);
                }
            }
            (Some("response_item"), "message") => {
                let role = payload.get("role").and_then(Value::as_str);
                for part in payload
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
                    match (role, part.get("type").and_then(Value::as_str)) {
                        (Some("assistant"), Some("output_text")) => transcript.text(text, at),
                        (Some("user"), Some("input_text")) if !events => {
                            let text = text.trim();
                            if !text.is_empty()
                                && !text.starts_with('<')
                                && !text.starts_with("# AGENTS.md")
                            {
                                transcript.prompt(text.to_string(), at);
                            }
                        }
                        _ => {}
                    }
                }
            }
            (Some("response_item"), "reasoning") => {
                let summary: Vec<&str> = payload
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|s| s.get("text").and_then(Value::as_str))
                    .collect();
                transcript.reasoning(&summary.join("\n\n"), at);
            }
            (Some("response_item"), "function_call") => transcript.tool(
                str_field(payload, "call_id").unwrap_or_default(),
                codex_function_call(
                    payload
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    payload
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ),
                at,
            ),
            (Some("response_item"), "custom_tool_call") => transcript.tool(
                str_field(payload, "call_id").unwrap_or_default(),
                codex_custom_call(
                    payload
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    payload
                        .get("input")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                ),
                at,
            ),
            (Some("response_item"), "local_shell_call") => transcript.tool(
                str_field(payload, "call_id")
                    .or_else(|| str_field(payload, "id"))
                    .unwrap_or_default(),
                ToolCall::Exec {
                    command: shell_words(payload.pointer("/action/command")),
                },
                at,
            ),
            (Some("response_item"), "web_search_call") => transcript.tool(
                str_field(payload, "id").unwrap_or_default(),
                ToolCall::WebSearch {
                    query: payload
                        .pointer("/action/query")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                },
                at,
            ),
            (Some("response_item"), "function_call_output" | "custom_tool_call_output") => {
                let id = str_field(payload, "call_id").unwrap_or_default();
                let failed = payload.get("output").is_some_and(codex_output_failed);
                transcript.tool_result(&id, failed, at);
            }
            _ => {}
        }
    })?;
    Ok(transcript)
}

fn codex_function_call(name: &str, arguments: &str) -> ToolCall {
    let args: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let text = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| args.get(*key))
            .map(|value| match value {
                Value::String(s) => s.clone(),
                other => shell_words(Some(other)),
            })
            .unwrap_or_default()
    };
    match name {
        "shell" | "container.exec" | "local_shell" => ToolCall::Exec {
            command: shell_words(args.get("command")),
        },
        "shell_command" | "exec_command" => ToolCall::Exec {
            command: text(&["command", "cmd"]),
        },
        "apply_patch" => ToolCall::ApplyPatch {
            path: patch_first_path(&text(&["input", "patch"])),
        },
        "update_plan" => ToolCall::Todo {
            items: args
                .get("plan")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|step| TodoItem {
                    text: step["step"].as_str().unwrap_or_default().to_string(),
                    done: step["status"].as_str() == Some("completed"),
                })
                .collect(),
        },
        "view_image" => ToolCall::ReadFile {
            path: text(&["path"]),
        },
        "web_search" | "search" => ToolCall::WebSearch {
            query: text(&["query", "q"]),
        },
        _ => match name.strip_prefix("mcp__").and_then(|r| r.split_once("__")) {
            Some((server, tool)) => ToolCall::Mcp {
                server: server.into(),
                tool: tool.into(),
                input: (!args.is_null()).then_some(args),
            },
            None => ToolCall::Unknown {
                name: name.into(),
                input: (!args.is_null()).then_some(args),
            },
        },
    }
}

fn codex_custom_call(name: &str, input: &str) -> ToolCall {
    match name {
        "apply_patch" => ToolCall::ApplyPatch {
            path: patch_first_path(input),
        },
        _ => ToolCall::Unknown {
            name: name.into(),
            input: None,
        },
    }
}

/// A shell argv as one command line; `bash -lc "<cmd>"`-style wrappers read
/// as the wrapped command.
fn shell_words(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    let words: Vec<&str> = value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    const WRAPPER_FLAGS: [&str; 5] = ["-lc", "-c", "-Command", "/c", "/C"];
    match words.as_slice() {
        [_, flag, command] if WRAPPER_FLAGS.contains(flag) => command.to_string(),
        _ => words.join(" "),
    }
}

/// The first file an `apply_patch` envelope touches.
fn patch_first_path(patch: &str) -> Option<String> {
    patch.lines().find_map(|line| {
        ["*** Update File: ", "*** Add File: ", "*** Delete File: "]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            .map(|path| path.trim().to_string())
    })
}

/// Codex reports failure as a non-zero exit code, either in a JSON envelope
/// (`metadata.exit_code`) or on the output's first line (`Exit code: N`).
fn codex_output_failed(output: &Value) -> bool {
    let text = match output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let trimmed = text.trim_start();
    if trimmed.starts_with('{')
        && let Ok(envelope) = serde_json::from_str::<Value>(trimmed)
        && let Some(code) = envelope
            .pointer("/metadata/exit_code")
            .and_then(Value::as_i64)
    {
        return code != 0;
    }
    trimmed
        .strip_prefix("Exit code: ")
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|code| code.parse::<i64>().ok())
        .is_some_and(|code| code != 0)
}

// ── opencode ────────────────────────────────────────────────────────────────

fn open_readonly(db: &Path) -> Option<rusqlite::Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = rusqlite::Connection::open_with_flags(db, flags).ok()?;
    let _ = conn.busy_timeout(Duration::from_millis(500));
    Some(conn)
}

/// Run a two-column text query into `out`; false when the query doesn't fit
/// this store's schema.
fn query_pairs(conn: &rusqlite::Connection, sql: &str, out: &mut HashMap<String, String>) -> bool {
    let Ok(mut stmt) = conn.prepare(sql) else {
        return false;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<String>>(1)?,
        ))
    }) else {
        return false;
    };
    for (key, value) in rows.flatten() {
        if let (Some(key), Some(value)) = (key, value) {
            out.insert(key, value);
        }
    }
    true
}

fn opencode_db(roots: &AgentStoreRoots) -> Option<rusqlite::Connection> {
    let db = roots.opencode_data.join("opencode.db");
    db.is_file().then(|| open_readonly(&db)).flatten()
}

fn opencode_sessions(roots: &AgentStoreRoots, project: &str) -> Vec<AgentSession> {
    let Some(conn) = opencode_db(roots) else {
        return Vec::new();
    };
    const PROMPTS: &str = "(SELECT COUNT(*) FROM message m WHERE m.session_id = s.id \
         AND json_extract(m.data, '$.role') = 'user')";
    let queries = [
        format!(
            "SELECT id, title, directory, time_updated, {PROMPTS} FROM session s \
             WHERE parent_id IS NULL AND time_archived IS NULL"
        ),
        format!("SELECT id, title, directory, time_updated, {PROMPTS} FROM session s"),
    ];
    for sql in &queries {
        let Ok(mut stmt) = conn.prepare(sql) else {
            continue;
        };
        let Ok(rows) = stmt.query_map([], |row| {
            Ok(AgentSession {
                id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                cwd: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                updated_at: row
                    .get::<_, Option<i64>>(3)?
                    .and_then(DateTime::from_timestamp_millis),
                prompt_count: row.get::<_, i64>(4)?.max(0) as usize,
            })
        }) else {
            continue;
        };
        return rows
            .flatten()
            .filter(|s| s.prompt_count > 0 && same_folder(&s.cwd, project))
            .map(|mut s| {
                if s.title.trim().is_empty() {
                    s.title = "Untitled conversation".into();
                }
                s
            })
            .collect();
    }
    Vec::new()
}

fn opencode_transcript(
    roots: &AgentStoreRoots,
    session_id: &str,
) -> Result<Transcript, EngineError> {
    let conn = opencode_db(roots).ok_or_else(|| not_found(session_id))?;
    let query_err = |err: rusqlite::Error| EngineError::Other(format!("opencode.db: {err}"));
    let mut transcript = Transcript::default();
    let (title, directory): (Option<String>, Option<String>) = conn
        .query_row(
            "SELECT title, directory FROM session WHERE id = ?1",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| not_found(session_id))?;
    transcript.title = title;
    transcript.cwd = directory;

    let mut parts: HashMap<String, Vec<Value>> = HashMap::new();
    let mut stmt = conn
        .prepare(
            "SELECT message_id, data FROM part WHERE session_id = ?1 ORDER BY time_created, id",
        )
        .map_err(query_err)?;
    let rows = stmt
        .query_map([session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(query_err)?;
    for (message_id, data) in rows.flatten() {
        if let Ok(part) = serde_json::from_str::<Value>(&data) {
            parts.entry(message_id).or_default().push(part);
        }
    }

    let mut stmt = conn
        .prepare("SELECT id, data FROM message WHERE session_id = ?1 ORDER BY time_created, id")
        .map_err(query_err)?;
    let rows = stmt
        .query_map([session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(query_err)?;
    for (message_id, data) in rows.flatten() {
        let Ok(message) = serde_json::from_str::<Value>(&data) else {
            continue;
        };
        let millis = |key: &str| {
            message
                .pointer(&format!("/time/{key}"))
                .and_then(Value::as_i64)
                .and_then(DateTime::from_timestamp_millis)
        };
        let (created, completed) = (millis("created"), millis("completed"));
        let message_parts = parts.remove(&message_id).unwrap_or_default();
        let flagged =
            |part: &Value, key: &str| part.get(key).and_then(Value::as_bool) == Some(true);
        match message.get("role").and_then(Value::as_str) {
            Some("user") => {
                let text: Vec<&str> = message_parts
                    .iter()
                    .filter(|p| {
                        p["type"] == "text" && !flagged(p, "synthetic") && !flagged(p, "ignored")
                    })
                    .filter_map(|p| p["text"].as_str())
                    .filter(|t| !t.trim().is_empty())
                    .collect();
                if !text.is_empty() {
                    transcript.prompt(text.join("\n\n"), created);
                }
            }
            Some("assistant") => {
                let at = completed.or(created);
                for part in &message_parts {
                    match part["type"].as_str() {
                        Some("text") if !flagged(part, "synthetic") => {
                            transcript.text(part["text"].as_str().unwrap_or_default(), at)
                        }
                        Some("reasoning") => {
                            transcript.reasoning(part["text"].as_str().unwrap_or_default(), at)
                        }
                        Some("tool") => {
                            let id = str_field(part, "callID")
                                .or_else(|| str_field(part, "id"))
                                .unwrap_or_default();
                            let name = part["tool"].as_str().unwrap_or_default();
                            transcript.tool(
                                id.clone(),
                                zeron_harness::opencode::decode_tool_call(
                                    name,
                                    &part["state"]["input"],
                                ),
                                at,
                            );
                            if part["state"]["status"] == "error" {
                                transcript.tool_result(&id, true, at);
                            }
                        }
                        _ => {}
                    }
                }
                transcript.current(at);
            }
            _ => {}
        }
    }
    Ok(transcript)
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

        fn project(&self, name: &str) -> String {
            let path = self.root.join("work").join(name);
            std::fs::create_dir_all(&path).unwrap();
            normalize_path(&path.to_string_lossy()).unwrap()
        }

        fn write(&self, rel: &str, lines: &[Value]) -> PathBuf {
            let path = self.root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
            std::fs::write(&path, body).unwrap();
            path
        }
    }

    fn claude_session(cwd: &str) -> Vec<Value> {
        vec![
            json!({"type": "queue-operation", "operation": "enqueue"}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:00:00Z",
                   "message": {"role": "user", "content": "Fix the failing test"}}),
            json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:02Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "thinking", "thinking": "Look at the test first."}]}}),
            json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:03Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "text", "text": "Running the tests."},
                       {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                        "input": {"command": "cargo test"}}]}}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:00:09Z",
                   "message": {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                        "content": "1 failed"}]}}),
            json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:10Z",
                   "isSidechain": true,
                   "message": {"role": "assistant", "content": [
                       {"type": "text", "text": "subagent chatter"}]}}),
            json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:20Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "tool_use", "id": "toolu_2", "name": "Edit",
                        "input": {"file_path": "src/lib.rs", "old_string": "a", "new_string": "b"}}]}}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:00:21Z",
                   "message": {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": "toolu_2", "content": "ok"}]}}),
            json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:30Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "text", "text": "Fixed it."}]}}),
            json!({"type": "user", "cwd": cwd, "isMeta": true, "timestamp": "2026-09-01T10:01:00Z",
                   "message": {"role": "user", "content": "<local-command-caveat>ignore</local-command-caveat>"}}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:01:01Z",
                   "message": {"role": "user", "content":
                       "<command-name>/model</command-name>\n<command-message>model</command-message>\n<command-args>opus</command-args>"}}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:01:02Z",
                   "message": {"role": "user", "content":
                       "<local-command-stdout>Set model</local-command-stdout>"}}),
            json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:02:00Z",
                   "message": {"role": "user", "content": [
                       {"type": "text", "text": "Now add docs<system-reminder>injected</system-reminder>"}]}}),
            json!({"type": "ai-title", "aiTitle": "Fix failing test", "sessionId": "s"}),
            json!({"type": "custom-title", "customTitle": "My test fix", "sessionId": "s"}),
        ]
    }

    fn texts(parts: &[MessagePart]) -> Vec<String> {
        parts
            .iter()
            .map(|part| match part {
                MessagePart::Text { text, .. } => format!("text:{text}"),
                MessagePart::Reasoning { text, .. } => format!("reasoning:{text}"),
                MessagePart::Tool {
                    call,
                    is_error,
                    resolved,
                    ..
                } => format!(
                    "tool:{}:{is_error}:{resolved}",
                    zeron_proto::view::tool_chip_content(call).0
                ),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn claude_transcript_becomes_turns_and_doc_entries() {
        let fx = Fixture::new();
        let cwd = fx.project("app");
        let file = fx.write("claude/projects/enc/abc-123.jsonl", &claude_session(&cwd));
        let transcript = claude_transcript(&file).unwrap();
        assert_eq!(transcript.title.as_deref(), Some("My test fix"));
        assert_eq!(transcript.cwd.as_deref(), Some(cwd.as_str()));
        let prompts: Vec<_> = transcript
            .turns
            .iter()
            .filter_map(|t| t.prompt.as_deref())
            .collect();
        assert_eq!(
            prompts,
            vec!["Fix the failing test", "/model opus", "Now add docs"],
            "meta lines, command output and reminders are not prompts"
        );

        let entries = doc_entries(&transcript, "dev-1");
        let roles: Vec<_> = entries.iter().map(|e| e.role).collect();
        assert_eq!(
            roles,
            vec![
                MessageRole::User,
                MessageRole::Assistant,
                MessageRole::User,
                MessageRole::User
            ]
        );
        assert_eq!(
            texts(&entries[1].parts),
            vec![
                "reasoning:Look at the test first.",
                "text:Running the tests.",
                "tool:Run:true:true",
                "tool:Edit:false:true",
                "text:Fixed it.",
            ],
            "sidechain lines are skipped; results resolve their calls"
        );
        match &entries[1].parts[3] {
            MessagePart::Tool { call, .. } => assert_eq!(
                call,
                &ToolCall::EditFile {
                    path: "src/lib.rs".into(),
                    old_string: None,
                    new_string: None
                },
                "edit strings are stripped like live transcripts"
            ),
            other => panic!("expected tool, got {other:?}"),
        }
        assert_eq!(entries[1].duration_ms, Some(30_000));
        assert!(
            entries
                .windows(2)
                .all(|w| w[0].created_at <= w[1].created_at)
        );
        assert!(
            entries
                .iter()
                .all(|e| e.status == Some(MessageStatus::Complete))
        );
    }

    #[test]
    fn claude_sessions_list_by_project_folder() {
        let fx = Fixture::new();
        let cwd = fx.project("app");
        let other = fx.project("other");
        let dir = format!("claude/projects/{}", claude_dir_name(&cwd));
        fx.write(&format!("{dir}/abc-123.jsonl"), &claude_session(&cwd));
        fx.write(
            &format!("{dir}/empty-1.jsonl"),
            &[json!({"type": "queue-operation"})],
        );
        fx.write(
            &format!("claude/projects/{}/zzz.jsonl", claude_dir_name(&other)),
            &claude_session(&other),
        );
        let sessions = list_sessions_with(&fx.roots, HarnessId::ClaudeCode, &cwd).unwrap();
        assert_eq!(sessions.len(), 1, "sessions without prompts are hidden");
        assert_eq!(sessions[0].id, "abc-123");
        assert_eq!(sessions[0].title, "My test fix");
        assert_eq!(sessions[0].prompt_count, 3);
        assert_eq!(
            claude_session_file(&fx.roots, "abc-123").unwrap(),
            fx.root.join(&dir).join("abc-123.jsonl")
        );
        assert!(load_transcript(&fx.roots, HarnessId::ClaudeCode, "../escape").is_err());
    }

    fn codex_rollout(cwd: &str) -> Vec<Value> {
        vec![
            json!({"timestamp": "2026-09-02T09:00:00Z", "type": "session_meta",
                   "payload": {"id": "0199-thread", "cwd": format!(r"\\?\{cwd}")}}),
            json!({"timestamp": "2026-09-02T09:00:01Z", "type": "response_item",
                   "payload": {"type": "message", "role": "user", "content": [
                       {"type": "input_text", "text": "<environment_context>x</environment_context>"}]}}),
            json!({"timestamp": "2026-09-02T09:00:01Z", "type": "event_msg",
                   "payload": {"type": "user_message", "message": "List the files"}}),
            json!({"timestamp": "2026-09-02T09:00:02Z", "type": "response_item",
                   "payload": {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Use ls."}]}}),
            json!({"timestamp": "2026-09-02T09:00:03Z", "type": "response_item",
                   "payload": {"type": "function_call", "name": "shell_command", "call_id": "c1",
                               "arguments": "{\"command\":\"ls -la\",\"workdir\":\".\"}"}}),
            json!({"timestamp": "2026-09-02T09:00:04Z", "type": "response_item",
                   "payload": {"type": "function_call_output", "call_id": "c1",
                               "output": "Exit code: 2\nWall time: 1s\nOutput:\nno such dir"}}),
            json!({"timestamp": "2026-09-02T09:00:05Z", "type": "response_item",
                   "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": "c2",
                               "input": "*** Begin Patch\n*** Update File: src/main.rs\n@@\n-a\n+b\n*** End Patch"}}),
            json!({"timestamp": "2026-09-02T09:00:06Z", "type": "response_item",
                   "payload": {"type": "custom_tool_call_output", "call_id": "c2",
                               "output": "Exit code: 0\nSuccess."}}),
            json!({"timestamp": "2026-09-02T09:00:07Z", "type": "response_item",
                   "payload": {"type": "message", "role": "assistant", "content": [
                       {"type": "output_text", "text": "Done."}]}}),
        ]
    }

    #[test]
    fn codex_rollouts_list_and_parse() {
        let fx = Fixture::new();
        let cwd = fx.project("svc");
        fx.write(
            "codex/sessions/2026/09/02/rollout-2026-09-02T09-00-00-0199-thread.jsonl",
            &codex_rollout(&cwd),
        );
        fx.write(
            "codex/session_index.jsonl",
            &[json!({"id": "0199-thread", "thread_name": "List files"})],
        );
        let sessions = list_sessions_with(&fx.roots, HarnessId::Codex, &cwd).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "0199-thread");
        assert_eq!(sessions[0].title, "List files");
        assert_eq!(sessions[0].cwd, cwd, "verbatim prefix stripped");
        assert_eq!(sessions[0].prompt_count, 1);

        let transcript = load_transcript(&fx.roots, HarnessId::Codex, "0199-thread").unwrap();
        let entries = doc_entries(&transcript, "dev-1");
        assert_eq!(entries.len(), 2, "injected context is not a prompt");
        assert_eq!(
            texts(&entries[1].parts),
            vec![
                "reasoning:Use ls.",
                "tool:Run:true:true",
                "tool:Patch:false:true",
                "text:Done."
            ]
        );
        match &entries[1].parts[2] {
            MessagePart::Tool { call, .. } => assert_eq!(
                call,
                &ToolCall::ApplyPatch {
                    path: Some("src/main.rs".into())
                }
            ),
            other => panic!("expected tool, got {other:?}"),
        }
    }

    #[test]
    fn codex_shell_argv_unwraps_wrappers() {
        assert_eq!(
            shell_words(Some(&json!(["bash", "-lc", "cargo test"]))),
            "cargo test"
        );
        assert_eq!(shell_words(Some(&json!(["git", "status"]))), "git status");
        assert!(codex_output_failed(&json!(
            "{\"output\":\"x\",\"metadata\":{\"exit_code\":1}}"
        )));
        assert!(!codex_output_failed(&json!("Exit code: 0\nok")));
    }

    #[test]
    fn opencode_sessions_list_and_parse() {
        let fx = Fixture::new();
        let cwd = fx.project("oc");
        std::fs::create_dir_all(&fx.roots.opencode_data).unwrap();
        let conn = rusqlite::Connection::open(fx.roots.opencode_data.join("opencode.db")).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT, parent_id TEXT, title TEXT, directory TEXT,
                                   time_updated INTEGER, time_archived INTEGER);
             CREATE TABLE message (id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
             CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES ('ses_1', NULL, 'Tidy imports', ?1, 1787849430040, NULL)",
            [&cwd],
        )
        .unwrap();
        let rows = [
            (
                "msg_1",
                1,
                json!({"role": "user", "time": {"created": 1787849400000_i64}}),
            ),
            (
                "msg_2",
                2,
                json!({"role": "assistant",
                                "time": {"created": 1787849401000_i64, "completed": 1787849430000_i64}}),
            ),
        ];
        for (id, at, data) in rows {
            conn.execute(
                "INSERT INTO message VALUES (?1, 'ses_1', ?2, ?3)",
                rusqlite::params![id, at, data.to_string()],
            )
            .unwrap();
        }
        let parts = [
            (
                "prt_1",
                "msg_1",
                json!({"type": "text", "text": "Tidy the imports"}),
            ),
            (
                "prt_2",
                "msg_1",
                json!({"type": "text", "text": "file dump", "synthetic": true}),
            ),
            (
                "prt_3",
                "msg_2",
                json!({"type": "reasoning", "text": "Check lib.rs"}),
            ),
            (
                "prt_4",
                "msg_2",
                json!({"type": "tool", "tool": "read", "callID": "call_1",
                                      "state": {"status": "completed", "input": {"filePath": "src/lib.rs"}}}),
            ),
            (
                "prt_5",
                "msg_2",
                json!({"type": "text", "text": "Imports tidied."}),
            ),
        ];
        for (index, (id, message, data)) in parts.into_iter().enumerate() {
            conn.execute(
                "INSERT INTO part VALUES (?1, ?2, 'ses_1', ?3, ?4)",
                rusqlite::params![id, message, index as i64, data.to_string()],
            )
            .unwrap();
        }
        drop(conn);

        let sessions = list_sessions_with(&fx.roots, HarnessId::Opencode, &cwd).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "Tidy imports");
        assert_eq!(sessions[0].prompt_count, 1);

        let transcript = load_transcript(&fx.roots, HarnessId::Opencode, "ses_1").unwrap();
        assert_eq!(transcript.first_prompt(), Some("Tidy the imports"));
        let entries = doc_entries(&transcript, "dev-1");
        assert_eq!(
            texts(&entries[1].parts),
            vec![
                "reasoning:Check lib.rs",
                "tool:Read:false:true",
                "text:Imports tidied."
            ]
        );
        assert_eq!(entries[1].duration_ms, Some(30_000));
    }

    #[test]
    fn preview_keeps_the_latest_messages() {
        let mut transcript = Transcript::default();
        for i in 0..100 {
            transcript.prompt(format!("prompt {i}"), None);
            transcript.text(&"x".repeat(PREVIEW_TEXT_MAX + 10), None);
            for n in 0..(PREVIEW_TOOLS_MAX + 2) {
                transcript.tool(
                    format!("t{i}-{n}"),
                    ToolCall::Exec {
                        command: "ls".into(),
                    },
                    None,
                );
            }
        }
        let preview = preview(&transcript);
        assert_eq!(preview.messages.len(), PREVIEW_MESSAGES);
        assert_eq!(preview.omitted, 200 - PREVIEW_MESSAGES);
        let last = preview.messages.last().unwrap();
        assert_eq!(last.role, AgentPreviewRole::Assistant);
        assert!(last.text.ends_with('…'));
        assert_eq!(last.tools.len(), PREVIEW_TOOLS_MAX + 1);
        assert_eq!(last.tools.last().unwrap(), "+2 more");
        assert_eq!(last.tools[0], "Run ls");
    }

    #[test]
    fn prompt_text_cleanup() {
        assert_eq!(claude_prompt_text("  hi  ").as_deref(), Some("hi"));
        assert_eq!(
            claude_prompt_text("<bash-input>ls</bash-input>").as_deref(),
            Some("! ls")
        );
        assert_eq!(claude_prompt_text("[Request interrupted by user]"), None);
        assert_eq!(
            claude_prompt_text("<system-reminder>only context</system-reminder>"),
            None
        );
        assert!(valid_session_id("0199d2-ab_c"));
        assert!(!valid_session_id("a/b"));
        assert!(!valid_session_id(""));
    }
}
