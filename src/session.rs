//! Model an inference session.
//!
//! This is also being used to attempt to drive some separation
//! between inference instantiation and subsequent interactions
//! while also hoping to extend some of the coherence for some
//! underlying concepts: a single configuration should produce
//! a session which can then inform subsequent logic.
//!
//! A wrinkle in the above is working through some of the
//! multi-threading concerns while also satisfying some of
//! related type constraints.
//!
//! This started off as cribbing from `ngrc`, but elides
//! the store construct to reduce the conceptual overhead.
//! Persistence could be added through some form of decoration.
//!
//! In the current state this could be folded into the `runtime`
//! module, but is left separate for future persistency.
//!
//! As implied below, this is used by the oi-kit StateMachine,
//! and specifically seems to inform some of the state transitions.
//! The specifics of this should be documented as it is discovered,
//! but the immediate takeaway is that session is logically
//! significant and should be kept in decent shape.
//!
//! This module is locally a leaf.
use oi_kit_agent::machine::MachineSession;
use oi_kit_providers::{
    base::Provider,
    conversation::{
        Conversation,
        message::Message,
    },
    model::ModelConfig,
};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Capture some of the more mutable data and align with some traits.
// Clone helps provides sane Mutex behavior.
#[derive(Clone)]
pub struct Session {
    /// Record the session and enable [MachineSession] use.
    conversation: Conversation,
    /// Record the identifier for the execution (to support [MachineSession]).
    pub id: String,
}
impl Session {
    /// Apply the provided `message` to the session.
    pub fn append_message(&mut self, message: Message) {
        self.conversation.push(message.with_generated_id_if_missing());
    }
}

/// Capture the execution environment.
///
/// This includes relatively immutable content,
/// along with the Session that requires additional controls.
///
/// This object itself should only provide methods that
/// do not introduce mutability.
///
/// The provide will use a dynamic reference to match
/// signatures in oi-kit
pub struct ExecutionEnvironment {
    /// Record the model configuration used for the session.
    model_config: ModelConfig,
    /// Provide access to the provider used for this execution.
    provider: Arc<dyn Provider>,
    /// Record ongoing session information.
    session: Arc<Mutex<Session>>,
}

impl ExecutionEnvironment {
    /// Initialize a session for the current environment,
    /// using the provided `provider` and `model_config` for inference.
    pub fn start_session<P: Provider + 'static>(provider: P, model_config: ModelConfig) -> Self {
        let id = Uuid::new_v4().to_string();
        ExecutionEnvironment {
            model_config,
            provider: Arc::new(provider),
            session: Arc::new(Mutex::new(Session {
                conversation: Conversation::empty(),
                id,
            })),
        }
    }

    /// Provide the configured model config.
    pub fn model_config(&self) -> ModelConfig {
        self.model_config.clone()
    }

    /// Provide the configured provider.
    pub fn provider(&self) -> Arc<dyn Provider> {
        self.provider.clone()
    }

    /// Produce the contained session.
    pub fn session(&self) -> Arc<Mutex<Session>> {
        Arc::clone(&self.session)
    }
}

/// Satisfy constraints for [oi_kit_agent::machine::StateMachine].
impl MachineSession for Session {
    fn id (&self) -> &str {
        &self.id
    }
    fn conversation(&self) -> Option<&Conversation> {
        Some(&self.conversation)
    }
}