/// Handle configuration parsing and validation.

// My go-to stance is to just make heavy use of environment variables...
// but there's so many knobs for agents that it seems like they'd
// have to be in a file anyway, so let's just use a file.

use anyhow::{anyhow, Result};
use configparser::ini::Ini;
use crate::hf;
use std::num::NonZero;
use std::path::Path;


pub struct Config {
    config: Ini,
}

impl Config {
    /// To start this will just return an HF Model.
    /// This should be swapped to an enum.
    pub fn model_info(&self) -> hf::HFModel {
        hf::HFModel {
            owner: self.config.get("model", "owner").unwrap(),
            repo: self.config.get("model", "repo").unwrap(),
            file: self.config.get("model", "file").unwrap(),
        }
    }

    pub fn context_size(&self) -> Option<NonZero<u32>> {
        self.config.getuint("context", "size").unwrap()
            .and_then(|size| NonZero::new(size as u32))

    }
}

/// Load configuration from the file at path.
pub fn from_path<T: AsRef<Path>>(path: T) -> Result<Config> {
    let mut config = Ini::new();
    // The upstream left case in this is a String.
    config.load(path).map_err(|e| anyhow!(e))?;
    Ok(Config { config })
}
