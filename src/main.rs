use anyhow::Result;
mod hf;

#[tokio::main]
async fn main() -> Result<()> {
    // init LLM
    let model = hf::HFModel {
        owner: "unsloth",
        repo: "Devstral-Small-2-24B-Instruct-2512-GGUF",
        file: "Devstral-Small-2-24B-Instruct-2512-UD-IQ1_M.gguf",
    };
    hf::path_for_hf_model(&model).await?;
    Ok(())
}
