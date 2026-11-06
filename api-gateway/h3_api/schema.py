from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Keyframe(StrictModel):
    asset_id: str = Field(min_length=1)
    time_seconds: float = Field(gt=0)


class VoiceReference(StrictModel):
    role: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)


class Lora(StrictModel):
    name: str = Field(min_length=1)
    strength: float = 1.0


class GenerateRequest(StrictModel):
    prompt: str = Field(min_length=1, description="Exact scene/camera/dialogue prompt; no automatic rewriting")
    avoid: str = Field(default="", description="Appended verbatim as avoid text, not a separate negative-conditioning channel")
    first_image: str | None = Field(default=None, description="Uploaded image asset ID")
    last_image: str | None = None
    reference_images: list[str] = Field(default_factory=list, max_length=9)
    keyframes: list[Keyframe] = Field(default_factory=list, max_length=7)
    drive_audio: str | None = Field(default=None, description="Uploaded driving/source audio asset; auto mode selects lock_source")
    final_audio: str | None = Field(default=None, description="Optional clean final soundtrack, only with lock_source; drive_audio still conditions video")
    voice_references: list[VoiceReference] = Field(default_factory=list, max_length=3, description="Ordered role/audio asset bindings for voice reference; target audio is generated unless lock_source")
    audio_mode: Literal["auto", "native", "reference_only", "lock_source", "remix_source"] = "auto"
    audio_denoise_strength: float = Field(default=0.35, ge=0, le=1)
    duration_seconds: float = Field(default=10, gt=0)
    resolution: Literal["480p", "720p", "1080p"] = "720p"
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4"] = "16:9"
    width: int | None = Field(default=None, ge=32, le=16384)
    height: int | None = Field(default=None, ge=32, le=16384)
    seed: int | None = Field(default=None, ge=0, le=18446744073709551615, description="Omit/null for a random 32-bit seed; actual seed is returned")
    steps: int = Field(default=8, ge=1, le=1000)
    model: str | None = None
    loras: list[Lora] | None = Field(default=None, description="Omit/null: profile defaults; []: no LoRA; supplied list replaces defaults")
    sampler: Literal["h3_turbo", "euler"] = "h3_turbo"
    scheduler: str = "simple"
    denoise: float = Field(default=1, gt=0, le=1)

    @field_validator("prompt")
    @classmethod
    def meaningful_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("first_image", "last_image", "drive_audio", "final_audio", "model")
    @classmethod
    def not_empty(cls, value):
        if value is not None and not value.strip():
            raise ValueError("omit absent inputs instead of sending empty strings")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if bool(self.width) != bool(self.height):
            raise ValueError("width and height must be provided together")
        if self.width and (self.width % 2 or self.height % 2):
            raise ValueError("MP4 output width and height must be even")
        if self.width and {"resolution", "aspect_ratio"} & self.model_fields_set:
            raise ValueError("choose width/height OR resolution/aspect_ratio, not both")
        if self.keyframes and not (self.first_image and self.last_image):
            raise ValueError("middle keyframes require both first_image and last_image in this timeline workflow")
        if len(self.reference_images) + len(self.keyframes) + bool(self.first_image) + bool(self.last_image) > 9:
            raise ValueError("H3 supports at most nine picture inputs in this workflow")
        if len({x.role for x in self.voice_references}) != len(self.voice_references):
            raise ValueError("voice reference role names must be unique")
        mode = self.resolved_audio_mode()
        if mode in {"lock_source", "remix_source"} and not self.drive_audio:
            raise ValueError(f"{mode} requires drive_audio")
        if mode == "reference_only" and not (self.drive_audio or self.voice_references):
            raise ValueError("reference_only requires drive_audio or voice_references")
        if self.final_audio and mode != "lock_source":
            raise ValueError("final_audio is only accepted with lock_source; generated audio must not be silently replaced")
        return self

    def resolved_audio_mode(self):
        if self.audio_mode != "auto":
            return self.audio_mode
        if self.drive_audio:
            return "lock_source"
        return "reference_only" if self.voice_references else "native"

