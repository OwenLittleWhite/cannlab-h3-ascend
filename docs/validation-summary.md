# Validation summary

This summarizes checks performed against the CANNLab A3 deployment. It deliberately excludes task IDs, user prompts, media, service URLs, and credentials.

## Runtime profile

- CANN 9.0.0; Python 3.11.4; PyTorch 2.7.1+cpu; torch_npu 2.7.1.post4.
- Two logical Ascend NPU devices were visible to the ComfyUI runtime.
- The validated smoke workload used 736×416, 124 frames, six Turbo steps, and the default Turbo v4 LoRA.

## API compatibility

- Regression coverage passed for all 23 request fields mapped from the existing H3 API contract.
- Audio paths exercised included voice references, driving audio, source-audio locking, final-audio selection, and source remixing.
- Request preview, validation, authentication, task status, and media download were checked on the live deployment.

## Transfer and loader correction

- CPU control-path checks cover transfer ordering, container preservation, and failure behavior.
- Dual-NPU checks exercised bit-exact tensor transfers and concurrent LoRA hook forwards.
- The actual H3 model/LoRA loader completed repeated two-device load/offload cycles with host-staged adapter transfers.
- The final transfer correction was not followed by a new end-to-end video render. Treat that acceptance item as pending.

## 2026-11-09 tokenizer backport

- The seven dedicated H3 special tokens were checked against their released IDs using the actual deployment tokenizer vocabulary on CPU.
- The installed ComfyUI H3 tokenizer's full text-tokenization entry point also returned the expected dedicated IDs for all seven markers on CPU.
- The original Ascend code outside the token map, registration subclass, and factory binding is unchanged by AST comparison.
- The existing audio schedule correction remains present; both conditioning-cache routes include the updated tokenizer source in their cache key.
- Service activation used an empty queue and preserved the job database. No video was generated as part of this update; audio-quality comparison remains pending.

## Reproduction boundary

These results apply to the pinned source and tested CANNLab image, not every Ascend model, driver, or NPU topology. The service was shut down after testing; this repository does not imply that a hosted endpoint is currently available.

