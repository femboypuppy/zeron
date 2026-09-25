//! Importing another agent's conversation: the chat row resumes the native
//! session from the conversation's own folder, the transcript is seeded on
//! chat2, the project is created on demand, and re-importing reopens the chat.

use std::path::Path;
use std::sync::Arc;

use serde_json::json;
use zeron_doc::{MessagePart, MessageRole};
use zeron_engine::agent_projects::AgentStoreRoots;
use zeron_engine::agent_sessions::import_session_with;
use zeron_engine::{EngineCore, EngineProfile, HarnessId, default_registry};
use zeron_proto::ImportedAgentSession;

fn roots(stores: &Path) -> AgentStoreRoots {
    AgentStoreRoots {
        claude_config: stores.join("claude"),
        claude_json: stores.join("claude.json"),
        codex_home: stores.join("codex"),
        opencode_data: stores.join("opencode"),
        pi_agent: stores.join("pi"),
        cursor_user: stores.join("cursor"),
        antigravity_user: stores.join("antigravity"),
        excluded: Vec::new(),
    }
}

fn write_claude_session(stores: &Path, cwd: &str, session_id: &str) {
    let dir = stores.join("claude").join("projects").join("encoded-app");
    std::fs::create_dir_all(&dir).unwrap();
    let lines = [
        json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:00:00Z",
               "message": {"role": "user", "content": "Fix the failing test"}}),
        json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:03Z",
               "message": {"role": "assistant", "content": [
                   {"type": "text", "text": "Running the tests."},
                   {"type": "tool_use", "id": "toolu_1", "name": "Bash",
                    "input": {"command": "cargo test"}}]}}),
        json!({"type": "user", "cwd": cwd, "timestamp": "2026-09-01T10:00:09Z",
               "message": {"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]}}),
        json!({"type": "assistant", "cwd": cwd, "timestamp": "2026-09-01T10:00:10Z",
               "message": {"role": "assistant", "content": [
                   {"type": "text", "text": "All green."}]}}),
    ];
    let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(dir.join(format!("{session_id}.jsonl")), body).unwrap();
}

/// The RPC's shape: the blocking import runs off the async runtime.
async fn import(core: &EngineCore, stores: &Path, session_id: &str) -> ImportedAgentSession {
    let (doc_host, workspace, roots, session_id) = (
        core.doc_host.clone(),
        core.workspace.clone(),
        roots(stores),
        session_id.to_string(),
    );
    tokio::task::spawn_blocking(move || {
        import_session_with(
            &doc_host,
            &workspace,
            &roots,
            HarnessId::ClaudeCode,
            &session_id,
        )
    })
    .await
    .unwrap()
    .expect("import")
}

#[tokio::test]
async fn claude_conversation_imports_as_a_resumable_chat() {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("work").join("app");
    std::fs::create_dir_all(&project).unwrap();
    let cwd = project.to_string_lossy().into_owned();
    let stores = dir.path().join("stores");
    write_claude_session(&stores, &cwd, "sess-1");

    let core = EngineCore::assemble_with_profile(
        EngineProfile::local(&dir.path().join("data")).unwrap(),
        Arc::new(default_registry()),
        HarnessId::Mock,
        None,
    )
    .unwrap();

    let imported = import(&core, &stores, "sess-1").await;
    assert!(!imported.existing);

    let chat = core
        .workspace
        .chat(&imported.chat_id)
        .unwrap()
        .expect("row");
    assert_eq!(chat.device_id, core.device_id);
    assert_eq!(chat.title.as_deref(), Some("Fix the failing test"));
    assert_eq!(chat.config.as_ref().unwrap().harness, HarnessId::ClaudeCode);
    assert_eq!(chat.room_gen, Some(2));
    assert_eq!(chat.space_id.as_deref(), Some(imported.space_id.as_str()));
    assert_eq!(chat.cwd.as_deref(), Some(cwd.as_str()));
    assert_eq!(
        core.workspace.chat_harness_session(&imported.chat_id),
        Some((
            "sess-1".to_string(),
            Some(cwd.clone()),
            Some(HarnessId::ClaudeCode)
        )),
        "the next run resumes the native session from the same cwd"
    );

    let spaces = core.workspace.read_spaces().unwrap();
    let space = spaces
        .iter()
        .find(|s| s.id == imported.space_id)
        .expect("project created");
    assert_eq!(space.device_id, core.device_id);
    assert_eq!(Path::new(&space.path), project.as_path());

    let entries = core
        .doc_host
        .open(&imported.chat_id)
        .unwrap()
        .doc()
        .read_entries()
        .unwrap();
    let roles: Vec<_> = entries.iter().map(|e| e.role).collect();
    assert_eq!(roles, vec![MessageRole::User, MessageRole::Assistant]);
    assert!(matches!(
        &entries[0].parts[..],
        [MessagePart::Text { text, .. }] if text == "Fix the failing test"
    ));
    assert_eq!(entries[1].parts.len(), 3, "text, tool, text");

    let again = import(&core, &stores, "sess-1").await;
    assert!(again.existing, "re-importing reopens the chat");
    assert_eq!(again.chat_id, imported.chat_id);

    // An existing project for the folder is reused rather than duplicated.
    write_claude_session(&stores, &cwd, "sess-2");
    let second = import(&core, &stores, "sess-2").await;
    assert_eq!(second.space_id, imported.space_id);
    assert_ne!(second.chat_id, imported.chat_id);

    core.shutdown().await;
}
