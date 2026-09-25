//! Switching a chat between agents: a stored session only resumes under the
//! agent that owns it; the first run under a new agent starts fresh with the
//! conversation handed over in its prompt (never in the transcript), and from
//! then on that agent's own session resumes.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

use zeron_doc::{MessageRole, MessageStatus, SessionCommandPayload, SessionMessageEntry};
use zeron_engine::{EngineCore, HarnessRegistry};
use zeron_harness::{Harness, HarnessError, RunControls};
use zeron_proto::{
    AgentEvent, DoneStatus, HarnessId, Model, ReasoningLevel, RunRequest, SandboxLevel,
    SteeringMode,
};

const CHAT: &str = "chat-switch";

type RequestLog = Arc<Mutex<Vec<(HarnessId, RunRequest)>>>;

/// Records every request under its own agent id and answers with a fixed
/// session id for that agent.
struct RecordingAgent {
    id: HarnessId,
    session_id: &'static str,
    requests: RequestLog,
}

#[async_trait]
impl Harness for RecordingAgent {
    fn id(&self) -> HarnessId {
        self.id
    }
    fn display_name(&self) -> &str {
        "Recording"
    }
    fn supports_steering(&self) -> bool {
        false
    }
    fn steering_mode(&self) -> SteeringMode {
        SteeringMode::TurnBoundary
    }
    fn reasoning_levels(&self) -> &[ReasoningLevel] {
        &[ReasoningLevel::Medium]
    }
    async fn models(&self) -> Result<Vec<Model>, HarnessError> {
        Ok(vec![])
    }
    async fn run(
        &self,
        request: RunRequest,
        _controls: RunControls,
    ) -> Result<BoxStream<'static, Result<AgentEvent, HarnessError>>, HarnessError> {
        self.requests
            .lock()
            .unwrap()
            .push((self.id, request.clone()));
        let events = vec![
            Ok(AgentEvent::SessionStarted {
                harness: self.id,
                model: "m".into(),
                tools: vec![],
                cwd: request.cwd.clone(),
                session_id: self.session_id.into(),
                assistant_message_id: "a".into(),
            }),
            Ok(AgentEvent::TextDelta {
                text: format!("{:?} got it", self.id),
            }),
            Ok(AgentEvent::Done {
                status: DoneStatus::Completed,
                result: None,
                error: None,
                session_id: Some(self.session_id.into()),
            }),
        ];
        Ok(futures::stream::iter(events).boxed())
    }
}

fn request(prompt: &str, harness: HarnessId) -> RunRequest {
    RunRequest {
        prompt: prompt.into(),
        harness: Some(harness),
        model: None,
        reasoning: None,
        model_options: Default::default(),
        cwd: "/tmp".into(),
        sandbox: SandboxLevel::WorkspaceWrite,
        auto_approve: true,
        attachments: Vec::new(),
        worktree: None,
        resume: None,
    }
}

fn entries(core: &EngineCore) -> Vec<SessionMessageEntry> {
    core.doc_host
        .open(CHAT)
        .ok()
        .and_then(|h| h.doc().read_entries().ok())
        .unwrap_or_default()
}

async fn turn(core: &EngineCore, prompt: &str, harness: HarnessId, message_id: &str) {
    let done_before = entries(core)
        .iter()
        .filter(|e| e.role == MessageRole::Assistant && e.status == Some(MessageStatus::Complete))
        .count();
    core.doc_host
        .queue_command(
            CHAT,
            SessionCommandPayload::Run {
                request: request(prompt, harness),
                message_id: message_id.into(),
            },
        )
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let done = entries(core)
            .iter()
            .filter(|e| {
                e.role == MessageRole::Assistant && e.status == Some(MessageStatus::Complete)
            })
            .count();
        if done > done_before {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "turn {message_id} timed out"
        );
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
}

#[tokio::test]
async fn switching_agents_hands_the_conversation_to_a_fresh_session() {
    let dir = tempfile::tempdir().unwrap();
    let requests: RequestLog = Arc::new(Mutex::new(Vec::new()));
    let registry = HarnessRegistry::new();
    for (id, session_id) in [
        (HarnessId::Mock, "mock-session"),
        (HarnessId::Codex, "codex-session"),
    ] {
        registry.register(Arc::new(RecordingAgent {
            id,
            session_id,
            requests: requests.clone(),
        }));
    }
    let core = EngineCore::assemble(dir.path(), Arc::new(registry), HarnessId::Mock, None).unwrap();
    core.workspace
        .create_space("space", &core.device_id, "/tmp", None, false)
        .unwrap();
    core.workspace
        .create_chat(CHAT, Some("space"), None, None, None)
        .unwrap();
    // Pre-titled: the auto-titler stays out of the request log.
    core.workspace.rename_chat(CHAT, "Switching").unwrap();

    turn(
        &core,
        "remember the codeword PINEAPPLE",
        HarnessId::Mock,
        "m1",
    )
    .await;
    assert_eq!(
        core.workspace.chat_harness_session(CHAT),
        Some((
            "mock-session".into(),
            Some("/tmp".into()),
            Some(HarnessId::Mock)
        )),
        "the stored session names its agent"
    );

    // Switch to another agent: no foreign resume, the history rides along.
    turn(&core, "what was the codeword?", HarnessId::Codex, "m2").await;
    {
        let log = requests.lock().unwrap();
        let (agent, second) = &log[1];
        assert_eq!(*agent, HarnessId::Codex);
        assert_eq!(
            second.resume, None,
            "a Mock session never resumes under Codex"
        );
        assert!(second.prompt.starts_with("<previous-conversation>"));
        assert!(second.prompt.contains("remember the codeword PINEAPPLE"));
        assert!(second.prompt.contains("Mock got it"));
        assert!(second.prompt.ends_with("what was the codeword?"));
    }
    let transcript = entries(&core);
    let typed: Vec<_> = transcript
        .iter()
        .filter(|e| e.role == MessageRole::User)
        .map(|e| match &e.parts[0] {
            zeron_doc::MessagePart::Text { text, .. } => text.clone(),
            other => panic!("unexpected part {other:?}"),
        })
        .collect();
    assert_eq!(
        typed,
        ["remember the codeword PINEAPPLE", "what was the codeword?"],
        "the transcript keeps the message as typed"
    );
    let row = core.workspace.chat(CHAT).unwrap().unwrap();
    assert!(
        !row.last_message_preview
            .unwrap_or_default()
            .contains("previous-conversation"),
        "the sidebar preview shows the typed message"
    );
    assert_eq!(
        core.workspace.chat_harness_session(CHAT).map(|s| s.2),
        Some(Some(HarnessId::Codex))
    );

    // Staying on the new agent resumes its own session, with no handoff.
    turn(&core, "and now?", HarnessId::Codex, "m3").await;
    {
        let log = requests.lock().unwrap();
        let (_, third) = &log[2];
        assert_eq!(third.resume.as_deref(), Some("codex-session"));
        assert_eq!(third.prompt, "and now?");
    }

    // Switching back hands over again (the old Mock session is not resumed).
    turn(&core, "back to you", HarnessId::Mock, "m4").await;
    {
        let log = requests.lock().unwrap();
        let (agent, fourth) = &log[3];
        assert_eq!(*agent, HarnessId::Mock);
        assert_eq!(fourth.resume, None);
        assert!(fourth.prompt.contains("was with Codex"));
        assert!(fourth.prompt.ends_with("back to you"));
    }
    core.shutdown().await;
}
