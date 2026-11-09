# H3 tokenizer fix and upstream compatibility

The deployed tokenizer now includes the missing H3 special-token registration from [ComfyUI PR #15808](https://github.com/Comfy-Org/ComfyUI/pull/15808). It retains the Ascend tensor-parallel and host-transfer changes in the existing tokenizer file. The updated file is shipped in `patches/comfyui-overlay/ComfyUI/comfy/text_encoders/minimax.py`; a standalone diff is in `patches/tokenizer-special-tokens-15808.patch`.

The [audio sampling correction in PR #15243](https://github.com/Comfy-Org/ComfyUI/pull/15243) was already present. Its ModelSamplingAV setup, audio latent scaling, and network velocity conversion were checked against the live source.

## Verified behavior

The CPU tokenizer check uses the deployment's real Qwen vocabulary and the candidate registration class without loading model weights. It verifies all seven token IDs (151669–151675), dialogue boundaries, inverse vocabulary, unchanged ordinary text, and preservation of the unrelated source AST. The check is provided in `validation/check_tokenizer_backport.py` and accepts `--source`, `--original`, `--tokenizer-dir`, and `--out` paths.

The actual installed `MiniMaxH3Tokenizer.tokenize_with_weights()` was also exercised on CPU for all seven markers. To repeat that check in a configured ComfyUI environment, run `python validation/check_comfy_tokenizer.py --comfy-root /path/to/ComfyUI --out /tmp/h3-tokenizer-check.json`. This loads tokenizer assets, not model weights.

Both persistent conditioning-cache routes fingerprint `comfy/text_encoders/minimax.py`, so the source change selects a new cache key. Existing cache files are retained. Activating the fix requires a clean ComfyUI process so an existing in-memory encoder is not reused.

The deployed service was switched only after the queue was empty. No generation job was submitted or replayed for this update. Token-ID and service checks do not establish end-to-end voice quality; a matched video/audio comparison remains pending.

## Official latest source review

The official `master` source was synchronized separately at commit `fca60fe70e5b84c5c6b058263653d15cb3401e44` (ComfyUI 0.44.0). The existing Ascend base patch fails to apply cleanly to that snapshot in the H3 model, model management, multi-device handling, operators, and other files. It has not replaced the deployed base.

The runtime remains pinned to `bd34f338ac505ea79e43968753968a464060e609` with the documented Ascend adaptation and local backports. A full 0.44.0 migration requires resolving and validating those changes separately.
