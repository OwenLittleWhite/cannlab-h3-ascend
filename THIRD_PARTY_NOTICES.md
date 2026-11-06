# Third-party notices

This project includes or patches upstream projects. Their copyrights and licenses remain with their respective authors.

| Component | Pinned source | License / handling |
| --- | --- | --- |
| ComfyUI base | `Comfy-Org/ComfyUI`, commit `bd34f338ac505ea79e43968753968a464060e609` | GPL-3.0; only the final overlay files are included here, not the full ComfyUI repository. |
| Ascend MiniMax-H3 adaptation | `Ascend-SACT/MiniMax-H3-Comfy-NPU`, commit `ac546057f925afafb4aea2597f0fca63763aa3db` | Its README declares Apache-2.0. The upstream patch is retained with its source reference; see `LICENSES/Apache-2.0.txt`. |
| MiniMax H3 Turbo ComfyUI node | `Larryvrh/ComfyUI-MiniMax-H3-Turbo`, commit `546b5028f4934f5129eb6c7142c2f3e461dfddbf` | Its upstream `LICENSE` is included beside the node source. Model weights are excluded. |
| Comfy API proxy | `Comfy-Org/comfy-api-proxy`, commit `1569dc17ec0dd293d4b9a0723e142204dacedb85` | MIT; vendored implementation is unmodified. The upstream license is included in both the vendored directory and `LICENSES/`. |
| Tensor serialization helper | `Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached`, commit `507c1ecd7c92439134e097e0fac34381340ab8c9` | MIT; copied helper and its license are under `custom_nodes/h3-ascend-cache/_vendor/`. |

The main project license does not replace the licenses of these components. Model checkpoints and LoRA weights are not redistributed by this repository; follow the model providers' terms when downloading them.

