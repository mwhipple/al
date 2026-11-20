# al

My AI assistant and general AI work.

## General Background

I have a general desire to have relatively in-depth understanding of
any agent/assistant I use - which is the same as any other tool
that I use though it seems particularly relevant in the current
state of AI where the tools are often products behind walled
gardens. This project will partly be driven by that exploration.

For similar reasons along with a penchant for distributed
execution, avoiding lock-in, and ethical concerns
(impacts to the environment and not feeding the data center
monster) I strongly prefer local inference.

### Tool Choice

My tool choice is largely driven by the above, alongside
the (unsurprising) notion that the killer feature of all of this agentic
stuff is the LLMs themselves, and that for the most part
the agentic workflows amount to message handlers and
providers on top of those LLMS. While so much of the current
AI tooling seems to amplify the tendency for systems to
get bigger and hungrier, this is specifically an area where
obfuscation and prospective lock-in feels likely to
undermine some of the potential benefits of LLMs
(powerful tools that can enable more rapid changes in
how we work shouldn't be buried in noise and rigidity).

#### llama.cpp

After playing with a few alternatives, I settled on
`llama.cpp` as my prefered LLM runtime. It seems particularly
conducive to coding against, and seems less likely
to "get in the way" than some of the others (I don't want
any models or knbos to be unnecessarily hidden).

One of the other notable benefits for llama.cpp is that
it provides a CLI intrface alongside a server: this
is particularly attractive for local inference as the
process can be neatly bound to when it's actually being
used, rather than being more oriented towards sucking
up resources to be ready when and if messages come in.
A server _could_ be spun up and tore down to provide
similar behavior while also providing some of the expected
APIs...but that feels a bit circuitous and anecdotally
things seem to run faster through the CLI.

#### Rust

As mentioned on my [website](https://www.mattwhipple.com),
I've recently broken down and become a Rustacean, and therefore
that will be the primarily used language. Certainly Python
or JavaScript may be more aligned with specific goals, but
honing my Rust skills should help with wider pursuits and for
this effort in particular also helps with...

#### Goose

Goose is the agent framework that I'm tentatively planning on
investing in. It provides a nice package to hit the ground
running, and I'll be digging in under the hood and either
making use of the GDK, or just getting up-to-speed on the
main applications.

Goose has relatively recently added support for local
`llama.cpp`, but _right now_ the support has fallen
a bit behind with a seemingly
[in-flight issue](https://github.com/aaif-goose/goose/issues/13402)
to get caught up again. That seems to be messing
with my ability to use newer models. Beyond that,
the performance seems dramatically worse when Goose
is in the mix versus using llama.cpp directly.
Ultimately this leads me to the idea that while I
tentatively want to adopt Goose, I also want to be ready
to work with it rather than just use it.

## Work

For now I'll track work in this doc...if I keep this going I may
split it out into some kind of structure.

### M1

The first pass is a basic one to get the ball rolling...
interact with llama.cpp using the Rust bindings
at https://github.com/utilityai/llama-cpp-rs
(also used by Goose).

#### Current State

This is currently working very roughly.
I'll be concurrently tidying this up and building out more of
a harness.

One glaring issue right now is that the model outputs become
quickly non-sensical which is plausibly a result of just
using default parameters.

## Laundry

- Support loading model info from parameter
- HF: Optionally provide notice about HuggingFace token
- HF: List model and maybe vary output for download vs. cache
      This could likely be determined by 0 total bytes
- Clean up streams/silence is golden