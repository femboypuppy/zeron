//! Rewinding a conversation: fork a chat at one of the user's messages.
//!
//! `ForkChat` creates a NEW chat holding the transcript up to (not including)
//! the chosen user message, and returns that message's text so the composer
//! can offer it again for editing, like Claude Code's rewind. The original
//! chat is untouched.
//!
//! An agent's native session can't be cut back to a mid-point, so the fork
//! starts with the "do not resume" tombstone. Its first run opens a fresh
//! session, and the engine hands it the forked transcript
//! ([`crate::handoff`]).

use chrono::Utc;
use zeron_doc::{MessagePart, MessageRole, MessageStatus, SessionDoc};
use zeron_proto::{Chat, ForkedChat};

use crate::EngineError;
use crate::doc_host::DocHost;
use crate::workspace_host::WorkspaceHost;

/// Fork `chat_id` just before its user message `message_id`.
pub fn fork_chat(
    doc_host: &DocHost,
    workspace: &WorkspaceHost,
    chat_id: &str,
    message_id: &str,
) -> Result<ForkedChat, EngineError> {
    let source = workspace
        .chat(chat_id)?
        .ok_or_else(|| EngineError::Other(format!("chat {chat_id} was not found")))?;
    if source.device_id != doc_host.device_id() {
        return Err(EngineError::Other(
            "a chat can only be rewound on the device that hosts it".into(),
        ));
    }
    let entries = doc_host.open(chat_id)?.doc().read_entries()?;
    let Some(at) = entries
        .iter()
        .position(|entry| entry.id == message_id && entry.role == MessageRole::User)
    else {
        return Err(EngineError::Other(format!(
            "message {message_id} is not one of this chat's user messages"
        )));
    };
    let prompt = entries[at]
        .parts
        .iter()
        .filter_map(|part| match part {
            MessagePart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let fork_id = crate::new_id();
    let doc = SessionDoc::init(&fork_id)?;
    for entry in &entries[..at] {
        let mut entry = entry.clone();
        // History is settled: nothing in the fork is still streaming.
        if entry.status == Some(MessageStatus::Streaming) {
            entry.status = Some(MessageStatus::Complete);
        }
        doc.push_message(&entry)?;
    }
    doc_host.seed_chat_snapshot(&fork_id, &doc.export_snapshot()?)?;

    let now = Utc::now();
    let title = source
        .title
        .as_deref()
        .filter(|title| !title.trim().is_empty())
        .map_or_else(
            || "Rewound session".to_string(),
            |t| format!("{t} (rewound)"),
        );
    workspace.import_chat_row(&Chat {
        id: fork_id.clone(),
        device_id: source.device_id.clone(),
        title: Some(title),
        archived: false,
        cwd: source.cwd.clone(),
        branch: source.branch.clone(),
        checkout_id: source.checkout_id.clone(),
        source_context: None,
        config: source.config.clone(),
        last_message_preview: None,
        last_message_at: Some(now),
        created_at: now,
        // Tombstone: never resume the source's session (it runs past the
        // rewind point); the first run starts fresh with the history handed
        // over.
        harness_session_id: Some(String::new()),
        harness_session_cwd: source.cwd.clone(),
        harness_session_harness: source.config.as_ref().map(|c| c.harness),
        space_id: source.space_id.clone(),
        last_seen_at: Some(now),
        room_gen: Some(2),
        parent_chat_id: None,
    })?;
    Ok(ForkedChat {
        chat_id: fork_id,
        space_id: source.space_id,
        prompt,
    })
}
