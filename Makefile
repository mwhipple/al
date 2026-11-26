CARGO = cargo

all: check cargo-run
.PHONY: all

cargo-%: ; $(CARGO) $(*)

check: cargo-check ;
.PHONY: check