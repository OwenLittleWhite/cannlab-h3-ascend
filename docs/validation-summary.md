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

## Reproduction boundary

These results apply to the pinned source and tested CANNLab image, not every Ascend model, driver, or NPU topology. The service was shut down after testing; this repository does not imply that a hosted endpoint is currently available.

