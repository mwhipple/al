//! Implement a composed Provider which delegates to specialists.
//!
//! The oi-kit [Provider] trait is effectively expected to handle
//! all model management, spanning management and use.
//! It seems valuable to be able to only select relevant
//! functionality, and prospectively reuse some strategies.

use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use oi_kit_providers::{
    base::{MessageStream, Provider},
    conversation::{
        message::Message,
    },
    errors::ProviderError,
    model::ModelConfig,
};
use rmcp::model::Tool;
use oi_kit_provider_types::conversation::token_usage::ProviderUsage;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

/// Dump out Messages as Strings for passing to a model.
///
/// This can provide some conventions such as OpenAI formatting.
/// see lib.rs in oi-kit local.
pub trait MessageStringifier {
    fn stringify(&self, req: &GenerationRequest<Self>) -> String where Self: Sized;
}

/// Provide a base stringifier that produces text with minimal formatting.
struct SimpleStringifier;
impl MessageStringifier for SimpleStringifier {
    fn stringify(&self, req: &GenerationRequest<SimpleStringifier>) -> String {
        req.messages.iter().map(Message::as_concat_text).collect::<Vec<_>>().join("\n")
    }
}

/// Model a request for inference.
/// This tentatively captures not only the data,
/// but some of the configuration details such as serialization.
pub struct GenerationRequest<'a, S: MessageStringifier> {
    /// Provide the context messages.
    messages: &'a [Message],
    /// Provide the strategy through which messages will be serialized to text.
    stringifier: S,
    /// Store the sender through which responses should be emitted.
    pub tx: &'a Sender<Result<(Option<Message>, Option<ProviderUsage>), ProviderError>>,
}
impl<'a, S: MessageStringifier> GenerationRequest<'a, S> {
    /// Produce the string representatino of this request.
    pub fn message_string(&self) -> String {
        self.stringifier.stringify(self)
    }
}

/// Define the Generator component.
///
/// This component should presume that it has been initialized,
/// and return an appropriate error if not.
///
/// Right now there is a single implementation, so this is strictly unnecessary.
/// The implementation itself will contain a lot of
/// specialized plumbing, so I'm expecting to split it
/// into a different modeule, and this serves to provide
/// desired directionality.
pub trait Generator: Send + Sync {
    fn generate<S: MessageStringifier>(&self, request: GenerationRequest<S>) -> Result<()>;
}

/// Define the DelegatingProvider struct.
///
/// This should evolve to provide general template style behavior,
/// where specifics are delegated to the struct members.
pub struct DelegatingProvider<G: Generator> {
    /// Name the provider for use in any analysis.
    pub name: String,
    /// Store the generator that will handle the inference requests.
    generator: Arc<G>,
}
impl<G: Generator> DelegatingProvider<G> {
    /// Instantiate a provider with the provided `name` and `generator`.
    pub fn create(name: &str, generator: G) -> Self {
        DelegatingProvider{name: name.to_string(), generator: Arc::new(generator)}
    }
}

#[async_trait]
impl<G: Generator + 'static> Provider for DelegatingProvider<G> {
    /// Return the stored name.
    fn get_name(&self) -> &str {
        &self.name
    }

    /// Stream the responses for the provided `messages`.
    ///
    /// This provides a lot of the supporting control flow and context propagation.
    /// This can also handle some of the underlying state management and verification.
    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<(Option<Message>, Option<ProviderUsage>), ProviderError>>(32);
        let generator = self.generator.clone();
        let messages = messages.to_vec();
        tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                    let request = GenerationRequest{messages: &messages, stringifier: SimpleStringifier, tx: &tx};
                    let _ = generator.generate(request);
                });

        });
        Ok(Box::pin(try_stream! {
            while let Some(result) = rx.recv().await {
                yield result?;
            }
        }))
    }
}