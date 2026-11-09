# MiniMax H3 on Ascend NPU / 昇腾部署适配

An open-source deployment adapter for running ComfyUI and MiniMax H3 on an Ascend NPU environment. It includes the CANNLab-tested custom nodes, API gateway, pinned ComfyUI overlays, and the transfer/cache fixes used by the deployment.

这是从一套实际运行过的 CANNLab A3 部署整理出的源码和补丁。它不是 ComfyUI、华为昇腾、CANNLab 或 MiniMax 的官方项目。

## What is included

- A persistent H3 conditioning cache and model lifecycle helper for repeated requests.
- An API gateway that maps the H3 request schema onto native ComfyUI workflows, including image references, voice references, driving audio, audio locking, and remix modes.
- Ascend-oriented ComfyUI overlays for multi-NPU coordination, text-encoder transfer, and safe LoRA tensor placement.
- Runtime launch/check scripts and CPU/NPU validation tests.
- Pinned upstream revisions and third-party notices.

The repository contains source code and configuration templates. It does not contain model weights, API tokens, user images, generated videos, runtime logs, or live service URLs.

The 2026-11-09 update backports upstream PR #15808, registering the seven H3 dialogue, lyrics, and caption tokens with their released IDs. The audio schedule correction from PR #15243 was already included in the pinned base. See [upstream update notes](docs/upstream-update-20261109.md).

## Validated environment

The deployment was validated on CANNLab A3 with two logical Ascend NPU devices exposed by a 910C instance, CANN 9.0.0, Python 3.11.4, PyTorch 2.7.1+cpu, and torch_npu 2.7.1.post4. The service used the platform's preinstalled CANN/PyTorch base and a persistent `/mnt/workspace` volume.

The 23-field API compatibility and audio-mode workflows were exercised on the deployment. The final LoRA-transfer correction passed CPU-path checks, dual-NPU tensor and hook checks, and real H3 model loading/offload cycles. A full video generation was not re-run after that last correction, so the final source snapshot should be treated as hardware-validated in components, with end-to-end validation still pending. More detail is in [validation notes](docs/validation-summary.md).

Hardware, CANN, torch_npu, and driver compatibility vary by image. Check the live device allocation with `npu-smi info` before using the example profile; do not assume that NPU device paths or the number of logical devices are fixed.

## Quick map

```text
custom_nodes/h3-ascend-cache/       persistent conditioning and health nodes
custom_nodes/ComfyUI-MiniMax-H3-Turbo/ pinned Turbo node source (weights excluded)
patches/Ascend-SACT/                 pinned base Ascend adaptation patch
patches/comfyui-overlay/             final deployment versions of patched ComfyUI files
api-gateway/                         authenticated API and native workflow compiler
deployment/scripts/                  runtime inspection and service launch scripts
tests/                               CPU, NPU, and model-loader checks
```

## Deployment outline

1. Start from an Ascend image with a matching, preinstalled CANN and `torch_npu` stack. Confirm the device allocation and persistent volume first.
2. Check out ComfyUI at the commit in `patches/Ascend-SACT/source_deps_info.json` and apply the pinned base patch in `patches/Ascend-SACT/`.
3. Copy the files under `patches/comfyui-overlay/` to the matching ComfyUI paths, then install the custom nodes from `custom_nodes/`.
4. Place the deployment under `/mnt/workspace/h3-ascend` or update the root constants in the launch scripts and custom node. Install model files separately using [the model manifest](docs/model-manifest.json); the weights are not redistributed here.
5. Copy `api-gateway/config/ascend.example.json` to `api-gateway/config/ascend.json`, set the live NPU device paths, and run the API install/launch scripts. The launcher creates a local bearer token with restrictive file permissions.

The scripts are specific to the tested CANNLab image and are not a general-purpose installer. Review and adapt the CANN environment path, device paths, and volume location for another image. Services bind to loopback by default. Do not expose them publicly without preserving bearer authentication and an authenticated HTTPS front end.

## License and upstream code

Original project code and ComfyUI modifications are released under GPL-3.0-only. Third-party components retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md) and `LICENSES/`.

