/// Implement support for retrieving models form HuggingFace.

/// This hopefully provides a more readily usable interface
/// on top of the hf_hub crate.

use hf_hub::{HFClient, HFError};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use std::{fmt::Write};
use std::path::PathBuf;

/// Identify a HuggingFace Model
pub struct HFModel<'a> {
        /// the repo owner. e.g. `TheBloke`
        pub owner: &'a str,
        /// the repo containing the model. e.g. `Llama-2-7B-Chat-GGUF`
        pub repo: &'a str,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        pub file: &'a str,
}

/// Return the local path for the provided model, downloading as needed.
pub async fn path_for_hf_model(model: &HFModel<'_>) -> Result<PathBuf, HFError> {
    HFClient::new()?
            .model(model.owner, model.repo)
            .download_file()
            .filename(model.file)
            .progress(console_handler())
            .send()
            .await
}

/// Produce a handler with a configured bar.
fn console_handler() -> ConsoleHandler {
    let bar = ProgressBar::no_length();
    bar.set_style(ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {binary_bytes}/{binary_total_bytes} {percent}% [ETA {eta_precise}] {msg}")
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("##-"));
    ConsoleHandler { bar }
}

/// Store the bar to share across callbacks.
// I foolishly wasted some time initializing the bar in the on_progress method
// since I deluded myself into thinking that each arm would be a separately
// defined callback that could access a shared closed variable.
struct ConsoleHandler {
    bar: ProgressBar,
}

/// Update a single progress bar.
impl ProgressHandler for ConsoleHandler {
    fn on_progress(&self, event: &ProgressEvent) {
        match event {
            ProgressEvent::Download(DownloadEvent::Start { total_bytes, total_files: _ }) => {
                    self.bar.set_length(total_bytes.clone());
            },
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                    let total_size = files.iter().map(|x| x.bytes_completed).sum::<u64>();
                    self.bar.set_position(total_size);
            },
            ProgressEvent::Download(DownloadEvent::Complete) => {
                    self.bar.finish();
            },
            _ => {},
        }
    }
}