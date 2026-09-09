# Hugging Face datasets used by the DFT task launchers

The project supports `math` and `offline_math` only.

`math` prepares the first 100,000 examples from `AI-MO/NuminaMath-CoT` and
uses Math500 for validation.

`offline_math` is not a fixed Hugging Face corpus. It generates four responses
per NuminaMath-CoT question with the base model, retains correct responses via
Math-Verify, and records the generator and source configuration in
`data/offline_math/manifest.json`. Regenerate it per base model with:

```bash
bash scripts/offline_math/generate_data.sh
```

The offline training launcher never generates data and rejects a dataset whose
manifest does not match its base model and generation configuration.
