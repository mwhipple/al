//! Provide the Runtime and supporting pieces to execute on top of oi-kit.
//!
//! A Runtime provides the top container for interacting with the agent.

/// Pull in session support.
use crate::session::{ExecutionEnvironment, Session};

use anyhow::{bail, Result};
use async_trait::async_trait;
use oi_kit_agent::{
    events::AgentEvent,
    inference::{InferenceEffect, InferenceRequestPreparer, InferenceRunner, PreparedInferenceRequest},
    machine::{EffectHandler, SessionLoader, StateMachine, Step},
    operation::{Emitter, InferenceInput, MachineEffect},
};
use oi_kit_providers::{
    base::Provider,
    conversation::{
        Conversation,
        message::Message,
        token_usage::ProviderUsage,
    },
    model::ModelConfig,
};
use std::sync::Arc;
use tokio::{
    sync::mpsc::UnboundedReceiver,
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

/// Provide some of the context within which a request will be evaluated.
///
/// This can presumably capture some background information and some
/// augmentation for user messages.
struct RequestContext {
    /// Provide the system prompt to pass to the model.
    system_prompt: String,
}
/// Project the context into a form that will be consumed by the model.
///
/// In the current incarnation this is just focused on the system_prompt.
#[async_trait]
impl InferenceRequestPreparer<Session> for RequestContext {
    async fn prepare(
        &self,
        _: &Session,
        _: &Conversation,
        input: InferenceInput,
    ) -> Result<PreparedInferenceRequest> {
        Ok(PreparedInferenceRequest {
            system_prompt: self.system_prompt.clone(),
            tools: input.tools,
            additional_messages: vec![],
        })
    }
}

/// Model the effects that will be emitted from the [StateMachine].
///
/// This is also used by the [InferenceRunner]
/// (and satsifies corresponding type constraints).
/// Thisre's some supporting documentation here:
/// <https://goose-docs.ai/docs/oi-kit/concepts/state-machine>
pub enum Effect {
    /// Represent a message within a conversation.
    Message(Message),
    /// Provide usage information.
    Usage(ProviderUsage),
}
/// Satisfy [InferenceEffect] dependency.
impl From<Message> for Effect {
    fn from(message: Message) -> Self {
        Self::Message(message)
    }
}
/// Support use with an [InferenceRunner].
impl InferenceEffect for Effect {
    fn record_usage(usage: ProviderUsage) -> Self {
        Self::Usage(usage)
    }
}
/// Support use with a [StateMachine].
///
/// This is unforuntate ex-post-facto validation :(.
impl MachineEffect for Effect {
    fn ensure_message_ids(&mut self) {
        if let Self::Message(message) = self {
            *message = message.clone().with_generated_id_if_missing();
        }
    }
}

/// Provide handlers for agent outputs.
///
/// This is a synched sink.
///
/// This trait is defined upfront to avoid a cycle,
/// even though there's only initially one implementation.
///
/// This is currently distributed using generics,
/// and is therefore resolved at compilation (which seems reasonable).
pub trait Sinker: Sync + Send {
    /// Produce a handler for the provided `rx`.
    fn sink(&self, rx: UnboundedReceiver<AgentEvent>) -> JoinHandle<Result<(), std::io::Error>>;
}

/// Contain and coordinate some of the runtime state.
pub struct Runtime<S: Sinker> {
    /// Distribute any cancellation requests.
    cancel: CancellationToken,
    /// Provide access to the session across threads.
    exec: ExecutionEnvironment,
    /// Store a strategy for producing an event sink.
    sinker: S,
}
impl<S: Sinker> Runtime<S> {
    /// Produce a runtime with the provided fields.
    pub fn create(
          cancel: CancellationToken,
          exec: ExecutionEnvironment,
          sinker: S,
    ) -> Self {
        Runtime { cancel, exec, sinker }
    }

    /// Produce a snapshot of the contained Session.
    fn session_snapshot(&self) -> Session {
        self.exec.session().lock().unwrap().clone()
    }

    /// Accept `prompt` text and process it as a user message.
    pub async fn process_from_user(&mut self, prompt: &str) -> Result<()> {
        let msg = Message::user().with_text(prompt);
        self.exec.append_message(msg);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let emitter = Emitter::new(tx, self.cancel.clone());
        let sink = self.sinker.sink(rx);
        let id = self.session_snapshot().id;
        let machine = agent(self.exec.provider(), self.exec.model_config(), self.cancel.clone());
        let _result = machine.run(self, &id, &emitter).await;
        drop(emitter);
        let _ = sink.await?;
        Ok(())
    }
}

/// Make Usable by the [StateMachine].
#[async_trait]
impl<S: Sinker> EffectHandler<Session, Effect> for Runtime<S> {
    async fn apply_effects(
        &self,
        session: &Session,
        _effects: &mut [Effect],
        _: &Emitter,
    ) -> Result<()> {
        if session.id != self.exec.session().lock().unwrap().id {
            bail!("effects belong to a different session");
        }
        Ok(())
    }
}

/// Make Usable by the [StateMachine].
#[async_trait]
impl<S: Sinker> SessionLoader<Session> for Runtime<S> {
    async fn load(&self, id: &str) -> Result<Session> {
        let session = self.session_snapshot();
        match session.id == id {
            true => Ok(session),
            false => bail!("unknown session {id}"),
        }
    }
}

/// This is a core part of prepping the runtime,
/// but also should likely be extensible.
fn agent(
     provider: Arc<dyn Provider>,
     model: ModelConfig,
     cancel: CancellationToken,
) -> StateMachine<'static, Session, Effect> {
    StateMachine::new(
        vec![
            Step::Inference(Arc::new(
            InferenceRunner::<Session, Effect>::new(provider, model)
                .with_request_preparer(Arc::new(RequestContext { system_prompt: "You are a helpful agent".to_string() })),
        )),
        ],
        cancel,
    )
}
