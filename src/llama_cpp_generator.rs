//! Implement a Generator that calls into llama.cpp.
//!
//! Compared to code in the oi-kit local inference crate,
//! this code works on focusing on model interactions only.
//! Some of the other concerns are almost certainly going
//! to creep back in somewhere or other, but the current goal
//! would be to keep those concerns relatively separate and
//! composable.
//!
//! This therefore focusees on the relatively basic need
//! of messages flowing to and from llama.cpp.

use crate::delegating_provider::{
    GenerationRequest,
    Generator,
    MessageStringifier,
};

use anyhow::Result;
use oi_kit_provider_types::{
    conversation::{
        message::Message,
        token_usage::ProviderUsage,
    },
    errors::ProviderError,
};

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    context::{
        LlamaContext,
        params::LlamaContextParams,
    },
    llama_batch::LlamaBatch,
    model::{
        LlamaModel,
        params::LlamaModelParams,
    },
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::{
    num::NonZero,
    path::PathBuf,
};
use tokio::sync::mpsc::Sender;

/// Enable interactions with (a configured) llama.cpp model.
pub struct LlamaCppGenerator {
    /// Store the backend for llama.cpp.
    ///
    /// I need to precisely identify what this is, but my initial
    /// conjecture is that it amounts to an execution environment.
    backend: LlamaBackend,
    /// Define the configuration that will be applied to created contexts.
    ctx_params: LlamaContextParams,
    /// Reference a loaded model.
    model: LlamaModel,
}
impl LlamaCppGenerator {

    /// Initialize the generator for accepting requests.
    ///
    /// This accepts a resolved `model_path` to a GGUF,
    /// and the `context_size` setting.
    /// `verbose` is also accepted here for the time being to reduce chatter,
    /// but should be part of a larger configuration class.
    pub fn initialize(
        model_path: PathBuf,
        context_size: Option<NonZero<u32>>,
        verbose: bool,
    ) -> Result<Self> {
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(verbose));
        let model_params = LlamaModelParams::default();
        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(context_size);
        Ok(LlamaCppGenerator { backend, model, ctx_params })
    }

    /// Apply [tokenization] to  `request`.
    ///
    /// [tokenization]: https://github.com/mwhipple/al/discussions/13
    fn tokenize_request<S: MessageStringifier>(&self, request: &GenerationRequest<S>) -> Vec<LlamaToken> {
        self.model.vocab().tokenize(request.message_string().as_bytes(), true, true)
    }

    /// Produce a context which is populated with prefilled tokens.
    ///
    /// This provides the initial state for the generation loop.
    fn create_and_prefill_context(&self, tokens: Vec<LlamaToken>) -> Result<LlamaContext<'_>, ProviderError> {
        let mut ctx = self.model
            .new_context(&self.backend, self.ctx_params.clone())
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to create context: {}", e)))?;

        let n_batch = ctx.n_batch() as usize;
        for chunk in tokens.chunks(n_batch) {
            let mut batch = LlamaBatch::get_one(chunk)
                .map_err(|e| ProviderError::ExecutionError(format!("Failed to create batch: {}", e)))?;
            ctx.decode(&mut batch)
                .map_err(|e| ProviderError::ExecutionError(format!("Failed to create batch: {}", e)))?;
        }

        Ok(ctx)
    }

    fn generation_loop(
        &self,
        mut ctx: LlamaContext<'_>,
        tx: &Sender<Result<(Option<Message>, Option<ProviderUsage>), ProviderError>>,
    ) -> Result<()> {
        let mut sampler = self.sampler();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let vocab = self.model.vocab();
        // Pull in max_output from inference_engine.rs
        for _ in 0..1_000_000 {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            // Maybe this isn't working?
            if vocab.is_eog(token) {
                break;
            }

            // Define what "piece" means here...maybe just a generic term for response part.
            let piece = vocab.token_to_piece(token, true, None);
            // Parsing the stream seems potentially useful...though that's not yet substantiated.
            let mut output_string = String::with_capacity(decoder.max_utf8_buffer_length(piece.len()).unwrap());
            let (_, _, _) = decoder.decode_to_string(&piece, &mut output_string, false);
            let message = Message::user().with_text(output_string);
            // That this is blocking can apparently cause crashes...
            // This should be encoded in the signatures for design time safety.
            tx.blocking_send(Ok((Some(message), None)))?;

            let next_tokens = [token];
            let mut next_batch = LlamaBatch::get_one(&next_tokens)
                .map_err(|e| ProviderError::ExecutionError(format!("Failed to create batch: {}", e)))?;
            ctx.decode(&mut next_batch)
                .map_err(|e| ProviderError::ExecutionError(format!("Decode failed: {}", e)))?;
        }
        Ok(())
    }

    /// Produce a sampler based on configuration.
    /// For now this will start with the canned sampler taken from the llama-cpp-rs sample,
    /// but should all be parameterized.
    fn sampler(&self) -> LlamaSampler {
        let seed = 1234;
        LlamaSampler::chain_simple([
            LlamaSampler::dist(seed),
            LlamaSampler::greedy(),
        ])
    }

}

/// Generation follows a pattern of prefilling the context,
/// then iteratively processing that context.
impl Generator for LlamaCppGenerator {
    fn generate<S: MessageStringifier>(&self, request: GenerationRequest<S>) -> Result<()> {
        let tokens = self.tokenize_request(&request);
        let ctx = self.create_and_prefill_context(tokens)?;
        Ok(self.generation_loop(ctx, request.tx)?)
    }
}
