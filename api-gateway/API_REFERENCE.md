# H3 音视频生成 API

This document describes the H3 API contract. Configure the API base URL and Bearer credential in a private client configuration; do not commit live credentials or deployment URLs.

当前入口使用 Bearer 鉴权；所有图片、音色参考和驱动音频先通过 `POST /v1/assets` 上传，再在生成请求里填返回的 `id`。通过 `POST /v1/preview` 检查实际提示词、素材编号、尺寸与帧数，通过 `POST /v1/validate` 检查素材和节点。`POST /v1/jobs` 提交生成必须带 `Idempotency-Key`。

服务检查：`GET /health` 表示外层 API 可达；带 Bearer 的 `GET /ready` 同时检查受管后端和真实生成线程。收到503时先检查服务状态，不要更换幂等键重复提交。`/docs` 为交互文档，`/openapi.json` 为接口定义。调用前先确认 `/ready` 为200。

## 音频模式

| audio_mode | 输入与输出行为 |
|---|---|
| `auto`（默认） | 有 `drive_audio` 时选择 `lock_source`；只有 `voice_references` 时选择 `reference_only`；均无时选择 `native` |
| `native` | 原生联合生成视频和音频。已提供的驱动音频/音色参考作为参考条件，不自动替换成片音轨 |
| `reference_only` | 驱动音频或音色样本只作为参考，目标音频由模型生成；不锁定目标音频 latent |
| `lock_source` | 必须提供 `drive_audio`。源音频编码进入目标 audio latent，音频噪声掩码为0；采样后按最大绝对误差≤1e-5校验 latent，并记录实际误差和是否逐位一致，成片使用 `final_audio`（如提供）或 `drive_audio` |
| `remix_source` | 必须提供 `drive_audio`。从源 audio latent 出发，由 `audio_denoise_strength`（0–1，默认0.35）控制重混强度；输出使用模型生成的音频 |

`voice_references` 最多3项，每项包含非空且不重复的 `role` 和音频 `asset_id`。接口为每个角色追加明确的 `<Audio N>` 音色参考绑定；原场景提示词保留。音色相似度、对白归属、口型与内容质量仍需验收，完成任务不等于这几项一定达到创作要求。

有 `drive_audio` 时，它是 `<Audio 1>`，音色参考从 `<Audio 2>` 开始；没有驱动音频时，音色参考从 `<Audio 1>` 开始。`final_audio` 只用于 `lock_source` 的最终音轨，不会替代用于驱动视频的 `drive_audio`，也不占参考音频编号。

驱动音频 latent 会按目标视频帧数裁剪或补零。`lock_source` 最终波形使用上传的原始驱动/最终音轨；请让音轨长度与 `applied_parameters.actual_duration_seconds` 接近，避免成片的音视频时长不匹配。源波形的完整性检查发生在有损 AAC 编码之前，不能声称 MP4 中的 AAC 字节与 WAV 字节相同。

## 图片与时间线

- `reference_images`：最多9个图片素材ID，用作身份/场景参考。
- `first_image` / `last_image`：视频第一帧与最后一帧的画面锚点。
- `keyframes`：最多7个 `{asset_id,time_seconds}`；需同时有首尾帧。按时间排序并对齐24fps帧索引，必须落在首尾帧之间，不能两个锚点占同一帧。
- 支持首尾帧、中间关键帧、身份图片与音频参考混用；所有图片合计最多9张。
- `<Picture N>` 编号依次为首帧、中间帧（时间顺序）、尾帧、身份参考图。`media_map` 和编译工作流返回实际对应关系。
- 本接口不接受参考视频。

## 生成参数

| 字段 | 说明 |
|---|---|
| `prompt` | 必填，非空；场景、镜头、角色、对白提示词 |
| `avoid` | 可选，原文追加到提示词，属于文字避让说明，不是独立负面条件通道 |
| `duration_seconds` | 默认10；按模型 `17*k+5` 帧网格向上对齐，固定24fps；实际帧数与时长会返回 |
| `resolution` | `480p` / `720p`（默认）/ `1080p` |
| `aspect_ratio` | `16:9`（默认）、`9:16`、`1:1`、`4:3`、`3:4` |
| `width`,`height` | 同时填写偶数值；与显式 `resolution` / `aspect_ratio` 二选一。内部画布向上补齐32，再中心裁剪到请求尺寸，不做输出放大 |
| `seed` | 可选；省略或null时随机，实际种子返回。换种子抽卡需换幂等键 |
| `steps` | 默认8，接受正整数，保留显式请求值。Turbo6/8与原生Euler20属于不同的采样选择 |
| `sampler` | `h3_turbo`（默认）或 `euler` |
| `scheduler` | 默认 `simple`；其他值须在 `GET /v1/models` 的后端可用列表中 |
| `denoise` | 大于0且不超过1，默认1 |
| `model` | 可选，使用已安装后端 checkpoint 名。省略时有图片/音频参考选Ref2VA，无参考选FL2VA；可用名见 `/v1/models` |
| `loras` | 省略/null使用默认Turbo v4 EMA；`[]`表示不加载LoRA；提供数组则替换默认，元素为 `{name,strength}`。支持组合与强度设置，重复文件会合并强度，避免覆盖注入 |

分辨率、时长、LoRA与采样器参数不再被旧的“仅720P、6–8步”适配限制拦截。可接受参数不代表所有极大尺寸都能装进当前设备：超出已验证工作负载会返回风险提示，实际运行也可能耗尽显存。不要把很短的高分辨率流程测试当成长视频的画质或内存验证。

## 请求例子

先上传图片和音频，替换下列占位ID：

```json
{
  "prompt": "<Picture 1>是成年男主，<Picture 2>是成年仙女。古风云海双人中景，男主说：今天天气真好。仙女回答：一起走走吧。自然普通话，无字幕。",
  "reference_images": ["HERO_IMAGE_ID", "FAIRY_IMAGE_ID"],
  "voice_references": [
    {"role": "男主", "asset_id": "MALE_VOICE_ID"},
    {"role": "仙女", "asset_id": "FEMALE_VOICE_ID"}
  ],
  "audio_mode": "reference_only",
  "duration_seconds": 5,
  "resolution": "720p",
  "steps": 8,
  "seed": 2026110501
}
```

驱动并锁定最终对白：在上例里增加 `drive_audio: "SOURCE_AUDIO_ID"`，将 `audio_mode` 改为 `lock_source`。如另有同时间轴的干净音轨，增加 `final_audio: "CLEAN_AUDIO_ID"`。为驱动音频分配 `<Audio 1>` 后，角色音色样本会自动绑定到 `<Audio 2>` / `<Audio 3>`。

重混源音轨：提供 `drive_audio`，选择 `audio_mode: "remix_source"`、`audio_denoise_strength: 0.25`，不提供 `final_audio`。

不使用加速 LoRA 的原生路线：`sampler: "euler"`、`steps: 20`、`loras: []`。这不会悄悄改成Turbo采样。

## 查询、下载与审计

- `GET /v1/jobs/{id}`：查询状态，在 `execution.outputs` 中按 `content_type: video/mp4` 查找结果URL。所有下载请求继续带 Bearer；完成前输出列表可能为空。
- `GET /v1/jobs/{id}/workflow`：查看实际编译工作流、解析后的参数与素材编号。
- `GET /v1/jobs/{id}/audit`：检查音频模式、源/目标 audio latent 哈希、源掩码、最终音轨来源、波形哈希与条件缓存命中。
- `POST /v1/jobs/{id}/cancel`：只请求取消这个任务，不使用全局中断兜底；需继续查询确认终态。
- 同一 `Idempotency-Key` + 相同请求返回原任务；同一键 + 不同参数返回409。传输超时先查询原任务，不要换键盲目重发。
- 暂时的公网 HTTPS 中转不支持 SSE，请轮询。实例内 SSE 保留。

所有条件缓存保存于持久化目录，指纹覆盖图片、参考音频、驱动音频、提示词、时间线、形状、模型身份和适配代码。缓存不会把一份音色或驱动音频误用到另一份输入上。代码/条件变更会重新建立缓存；同条件换种子、换步数可以复用。

`cache_status` 记录条件节点最近一次执行时的磁盘缓存状态；若 ComfyUI 直接复用内存中的条件节点，沿用该报告，不能用它单独计算本任务耗时。
