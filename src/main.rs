use anyhow::{Context, Result};
use encoding_rs::Decoder;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::env;
use std::io::Write;

mod cfg;
mod hf;

#[tokio::main]
async fn main() -> Result<()> {
    // Resolve the model file - the path should be loaded from an envvar.
    let cfg_path = env::var("AL_CONFIG_FILE").context("AL_CONFIG_FILE must be set in the environment")?;
    let cfg = cfg::from_path(cfg_path)?;
    let model_info = cfg.model_info();
    let model_path = hf::path_for_hf_model(&model_info).await?;
    let model_params = LlamaModelParams::default();

    // Load the model.
    let backend = LlamaBackend::init()?;
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)?;

    // Initialize the context.
    let ctx_params = LlamaContextParams::default().with_n_ctx(cfg.context_size());
    let mut ctx = model.new_context(&backend, ctx_params)?;

    // Deal with a prompt.
    let prompt = "Hello, can you tell me what time it is?";
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
    let mut n_cur = batch.n_tokens();
    let n_len = 1_000_000;

    // It seems like this could be cleanly split out along the lines of
    // a runner and a stepper.
    let mut stepper = BatchStepper { sampler, ctx, batch, model: &model, decoder };
    stepper.decode();
    while n_cur <= n_len {
        // Sample next token.
        {
            if stepper.advance(n_cur)? { break };
        }
        n_cur += 1;
        stepper.decode();
    }
    Ok(())
}

/// Support processing of batches.
// This is a rough initial pass at containing the data -
// this needs a fair amount of refinment.

struct BatchStepper<'a> {
    sampler: LlamaSampler,
    ctx: LlamaContext<'a>,
    batch: LlamaBatch<'a>,
    model: &'a LlamaModel,
    decoder: Decoder,
}

impl BatchStepper<'_> {
    // The boolean return is a bit of a kludge - it probably makes sense to
    // return some kind of state enum which could also help with the control flow.
    fn advance(&mut self, n_cur: i32) -> Result<bool> {
        let token = self.sampler.sample(&self.ctx, self.batch.n_tokens() - 1);
        self.sampler.accept(token);

        // is it an end of stream?
        if self.model.vocab().is_eog(token) {
            eprintln!();
            return Ok(true)
        }
        let tokens = self.model.vocab().token_to_piece(token, true, None);
        let mut output_string = String::with_capacity(self.decoder.max_utf8_buffer_length(tokens.len()).unwrap());
        let (_, _, _) = self.decoder.decode_to_string(&tokens, &mut output_string, false);
        print!("{output_string}");
        std::io::stdout().flush()?;

        // Advance the batch.
        self.batch.clear();
        self.batch.add(token, n_cur, &[0], true)?;

        Ok(false)
    }

    fn decode(&mut self) {
        self.ctx.decode(&mut self.batch);
    }
}


