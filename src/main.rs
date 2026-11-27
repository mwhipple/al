//! al is my personal AI assistant.
//!
//! My focus is primarily on building around local inference, where I'm
//! initially expecting to use [llama.cpp] and [oi-kit].
//! This was additionally started by stealing from [llama-cpp-rs] and [ngrc].
//!
//! I have some reservations about some of the constructs currently provided
//! in oi-kit...so I'll be using some and exploring alternatives to others.
//! The impetus for some of the concerns is that support for
//! llama.cpp in oi-kit is lagging, and there are a lot of pieces
//! that surround that support, so patching isn't particularly quick.
//! This extends to a few themes in the some of the current design decisions
//! which complicate the ability to decompose the project.
//!
//! The first is that model management and use seem to be bundled together -
//! while both seem valuable it seems like it would be beneficial to
//! have a distinct boundary between interacting with an ostensibly
//! running model and the various details of launching those models
//! (where the former could also benefit from more abstraction).
//!
//! The second is somewhat around the wider design and is driven
//! by half-baked ideas which could definitely turn out to be bad.
//! From earlier basic efforts and supported by the oi-kit
//! [StateMachine](https://goose-docs.ai/docs/oi-kit/concepts/state-machine)
//! model it seems like a particularly tidy way to approach
//! an agent harness is in terms of a fairly general
//! orchestration framework on top of model interactions.
//! An extension of this is that virtually all of the pieces such as
//! tools can be plugged into those general ports rather than expanding
//! the core conceptual footprint.
//!
//! These ideas are certainly subjective and too early on to trust.
//! oi-kit and goose provide far more functionality than what is here
//! or what is likely to be here unless this gains additional
//! stakeholders, since I'm personally more likely to just focus
//! on the small subset of behaviors that I need.
//!
//! This project will be built unapolagetically with [tokio],
//! but to mirror the theft sources, and also given that it
//! it seems to somewhat be a de facto standard in the Rust world.
//!
//! [llama.cpp]: https://llama-cpp.com/
//! [llama-cpp-rs]: https://github.com/utilityai/llama-cpp-rs
//! [ngrc]: https://github.com/jamadeo/ngrc
//! [oi-kit]: https://goose-docs.ai/docs/oi-kit
//! [tokio]: https://tokio.rs/
use anyhow::{Context, Result};

use oi_kit_providers::model::ModelConfig;
use std::env;
use tokio_util::sync::CancellationToken;

mod cfg;
mod cli;
mod hf;
mod delegating_provider;
mod llama_cpp_generator;
mod runtime;
mod session;

/// Launch al.
///
/// This should remain focused on the wiring of other components,
/// rather than containing any significant logic.
///
/// This primarily involves instantiating a [runtime] and
/// [UI](cli).

#[tokio::main]
async fn main() -> Result<()> {
    let cfg_path = env::var("AL_CONFIG_FILE").context("AL_CONFIG_FILE must be set in the environment")?;
    let cfg = cfg::Config::from_path(cfg_path)?;
    let model_info = cfg.model_info();
    let model_path = hf::path_for_hf_model(&model_info).await?;

    let cancel = CancellationToken::new();

    let model_config = ModelConfig::new(model_info.to_string());
    let generator = llama_cpp_generator::LlamaCppGenerator::initialize(
        model_path,
        cfg.context_size(),
        false,
    )?;

    let provider = delegating_provider::DelegatingProvider::create("Test", generator);

    let runtime = runtime::Runtime::create(
        cancel.clone(),
        session::ExecutionEnvironment::start_session(provider, model_config),
        cli::Rendererer,
        cfg.system_prompt().unwrap_or("You are a helpful assistant".to_string()),
    );
    cli::cli_input(runtime, cancel.clone()).await?;
    Ok(())
}
