use anyhow::Result;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::io::Write;

mod hf;

#[tokio::main]
async fn main() -> Result<()> {
    // Resolve the model file.
    let model_info = hf::HFModel {
        owner: "unsloth",
        repo: "Devstral-Small-2-24B-Instruct-2512-GGUF",
        file: "Devstral-Small-2-24B-Instruct-2512-UD-IQ1_M.gguf",
    };
    let model_path = hf::path_for_hf_model(&model_info).await?;
    let model_params = LlamaModelParams::default();

    // Load the model.
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    // Initialize the context.
    let ctx_params = LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Deal with a prompt.
    let prompt = "Hello, Al!";
    let tokens_list = model.vocab().tokenize(prompt.as_bytes(), true, true);
    // Check kv cache...skipping for now.

    // Create a llama_batch to submit token data for decoding.
    // The batch parameters should be parameterized.
    // Load the tokens and output the logit for the last one.
    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        batch.add(token, i, &[0], i == last_index)?;
    }

    // Sampling seems to be the front door provided by llama.cpp.
    // This should be parameterized.
    let seed = 1234;
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed),
        LlamaSampler::greedy(),
    ]);

    // Prepare to render results.
    let mut decoder = encoding_rs::UTF_8.new_decoder();


    // Populate and submit the context.
    ctx.decode(&mut batch)?;

    let mut n_cur = batch.n_tokens();
    let n_len = 1_000_000;

    while n_cur <= n_len {
        // Sample next token.
        {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // is it an end of stream?
            if model.vocab().is_eog(token) {
                eprintln!();
                break;
            }

            let tokens = model.vocab().token_to_piece(token, true, None);
            let mut output_string = String::with_capacity(decoder.max_utf8_buffer_length(tokens.len()).unwrap());
            let (_, _, _) = decoder.decode_to_string(&tokens, &mut output_string, false);
            print!("{output_string}");
            std::io::stdout().flush()?;

            // Advance the batch.
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
        }

        n_cur += 1;
        ctx.decode(&mut batch)?;
    }
    Ok(())
}
