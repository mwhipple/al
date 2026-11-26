//! Provide CLI UI.
//!
//! The means to interface with the system (Views/UIs) are
//! split out into the concept of UI. This makes use of the
//! concepts of Inputs and Sinks.
//!
//! UIs leverage compile-time polymorphism which seems sufficient.
//!
//! ## Input
//!
//! Inputs tend to be related to the main loops of programs,
//! and it therefore seems straightforward to have them
//! call into the system rather than expecting to be called
//! (reversion of control?).
//! Inputs therefore only exist as a concept only, but are
//! not more formally modeled.
//!
//! ## Sink
//!
//! A Sink receives output from the system, and therefore
//! needs to be attached somewhere to be called.
//! The trait is currently defined in [runtime] but documented
//! here for cohesion with Input and to be alongside
//! the active implementation.
//!
//! The trait itself is [Sinker] which produces Sinks,
//! which is stolen from oi-kit/ngrc. The specifics needs
//! related to object lifespans will need further investigation,
//! but this contract should satisfy more cases regardless.
//!
//! Sink is used as a general term given that the underlying
//! contract is indifferent to how the outputs are used
//! (e.g., whether they are displayed), but more specific
//! terms may be used, such as [Rendererer] below.
use crate::runtime::{Runtime, Sinker};

use oi_kit_providers::conversation::message::MessageContent;
use oi_kit_agent::events::AgentEvent;
use std::io::{self, BufRead, Write, Error};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Produce an input driven by the CLI
/// which will call `runtime` and respect `cancel`.
///
/// This currently makes use of standard streams
/// (which could presumably be redirected).
///
/// The specific uses of stdin/stdout/stderr should be
/// clearly elucidated.
pub async fn cli_input<S:Sinker>(
    mut runtime: Runtime<S>,
    cancel: CancellationToken,
) -> Result<(), Error> {
    let mut lines = stdin_lines();
    loop {
        if cancel.is_cancelled() {
            break;
        }

        // Retrieve prompt (or cancel).
        eprint!("\n> ");
        io::stderr().flush()?;
        let prompt = match tokio::select! {
            biased; // prefer options in order listed.
            _ = cancel.cancelled() => break,
            line = lines.recv() => line,
        } {
            Some(line) => line?,
            None => break,
        };
        // Skip empty prompts.
        if prompt.trim().is_empty() {
            continue;
        }
        let _ = runtime.process_from_user(&prompt).await;
     }
    Ok(())
}

/// Allow idle Ctrl-C exit without blocking the stdin task.
///
/// Only one line is queued ahead of the current turn.
fn stdin_lines() -> tokio::sync::mpsc::Receiver<io::Result<String>> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let failed = line.is_err();
            if tx.blocking_send(line).is_err() || failed {
                break;
            }
        }
   });
   rx
}

/// Provide a [Sinker] to render output in the CLI.
///
/// A Rendererer produces a Renderer.
pub struct Renderer;

impl Sinker for Renderer {
    fn sink(
        &self,
        mut rx: UnboundedReceiver<AgentEvent>,
    ) -> JoinHandle<Result<(), Error>> {
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if let AgentEvent::Message(message) = event {
                    for content in message.content {
                        match content {
                            MessageContent::Text(text) => {
                                print!("{}", text.text);
                                io::stdout().flush()?;
                            }
                            MessageContent::Error(error) => eprintln!("\n{}", error.message),
                            MessageContent::SystemNotification(note) => eprintln!("\n{}", note.msg),
                           _ => {},
                        }
                    }
                }
           }
           println!();
           Ok::<_, io::Error>(())
       })
    }
}
