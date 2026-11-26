//! Handle configuration parsing and validation.
//!
//! While I typically gravitate towards environment variables,
//! there's enough going on with agents to prefer a more package
//! friendly alternative, so instead this will make use of ini files.
//!
//! Usage will primarily revolve around instantation of
//! a [Config] and then calling appropriate methods on that instance.

use anyhow::{anyhow, Result};
use configparser::ini::Ini;

/// This module pulls in [hf] to support model retrieval.
use crate::hf;
use std::num::NonZero;
use std::path::Path;

/// Provide access to the current configuration.
pub struct Config {
    /// Store the raw loaded configuration.
    config: Ini,
}

impl Config {

    /// Load configuration from the file at `path`.
    pub fn from_path<T: AsRef<Path>>(path: T) -> Result<Config> {
        let mut config = Ini::new();
        // The upstream left case in this is a String.
        config.load(path).map_err(|e| anyhow!(e))?;
        Ok(Config { config })
    }

    /// Identify the model to use for inference.
    ///
    /// To start this will just return an [hf::HFModel].
    /// This should be swapped to an enum.
    pub fn model_info(&self) -> hf::HFModel {
        hf::HFModel {
            owner: self.config.get("model", "owner").unwrap(),
            repo: self.config.get("model", "repo").unwrap(),
            file: self.config.get("model", "file").unwrap(),
        }
    }

    /// Return the context size to allocate.
    pub fn context_size(&self) -> Option<NonZero<u32>> {
        self.config.getuint("context", "size").unwrap()
            .and_then(|size| NonZero::new(size as u32))

    }
}
