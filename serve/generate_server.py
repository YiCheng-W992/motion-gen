import argparse
import copy
import io
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None

from data_loaders.tensors import collate
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_diffusion_options,
    add_generate_options,
    add_model_options,
    add_sampling_options,
    apply_rules,
    get_args_per_group_name,
    get_cond_mode,
    load_args_from_model,
)
from utils.sampler_util import ClassifierFreeSampleModel


class GenerateRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=200,
        description="Text prompt to generate a motion (max 200 characters).",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


@dataclass
class ServerState:
    args: argparse.Namespace
    model: torch.nn.Module
    guided_model: Optional[torch.nn.Module]
    diffusion: object
    mean: np.ndarray
    std: np.ndarray
    max_frames: int
    fps: float
    gen_steps: int
    lock: threading.Lock


class _DummyData:
    dataset = type("Dataset", (), {})()


def _build_args(model_path: str, device: int, use_ema_override: Optional[bool]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args(["--model_path", model_path, "--device", str(device)])

    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args = load_args_from_model(args, args_to_overwrite)

    if use_ema_override is not None:
        args.use_ema = use_ema_override
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return apply_rules(args)


def _resolve_data_root(args: argparse.Namespace) -> str:
    if getattr(args, "data_dir", ""):
        return args.data_dir
    return "./dataset/HumanML3D"


def _init_state() -> ServerState:
    model_path = os.environ.get("MDM_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("MDM_MODEL_PATH is required.")
    device = int(os.environ.get("MDM_DEVICE", "0"))
    use_ema_env = os.environ.get("MDM_USE_EMA")
    use_ema = None
    if use_ema_env is not None:
        use_ema = use_ema_env.lower() in ("1", "true", "yes")
    gen_steps = int(os.environ.get("MDM_SAMPLE_STEPS", "100"))

    args = _build_args(model_path, device, use_ema)
    dist_util.setup_dist(args.device)

    if args.dataset != "humanml":
        raise RuntimeError(f"Only HumanML is supported, got dataset={args.dataset!r}.")
    if args.arch != "trans_dec":
        raise RuntimeError("Only trans_dec is supported.")
    if args.text_encoder_type != "bert":
        raise RuntimeError("Only BERT text encoder is supported.")
    if get_cond_mode(args) != "text":
        raise RuntimeError("Only text-conditioned models are supported.")

    max_frames = 196
    fps = 20

    model, diffusion = create_model_and_diffusion(args, _DummyData())
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model.to(dist_util.dev())
    model.eval()

    guided_model = None
    if args.guidance_param != 1:
        if model.cond_mask_prob <= 0:
            raise RuntimeError("Guidance requested but checkpoint was not trained with cond_mask_prob > 0.")
        guided_model = ClassifierFreeSampleModel(model)
        guided_model.to(dist_util.dev())
        guided_model.eval()

    data_root = _resolve_data_root(args)
    mean = np.load(os.path.join(data_root, "Mean.npy"))
    std = np.load(os.path.join(data_root, "Std.npy"))

    return ServerState(
        args=args,
        model=model,
        guided_model=guided_model,
        diffusion=diffusion,
        mean=mean,
        std=std,
        max_frames=max_frames,
        fps=fps,
        gen_steps=gen_steps,
        lock=threading.Lock(),
    )


def _sample(state: ServerState, req: GenerateRequest) -> bytes:
    args = copy.deepcopy(state.args)
    prompt = req.text.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="text must be non-empty.")
    if len(prompt) > 200:
        raise HTTPException(status_code=422, detail="text must be at most 200 characters.")

    args.num_samples = 1
    args.num_repetitions = 1

    n_frames = min(state.max_frames, int(args.motion_length * state.fps))
    if n_frames <= 0:
        raise HTTPException(status_code=400, detail="motion_length too small for current FPS.")

    args.batch_size = args.num_samples
    texts = [prompt] * args.num_samples

    model_to_use = state.model
    if args.guidance_param != 1:
        if state.guided_model is None:
            raise HTTPException(status_code=400, detail="Guidance is not available for this checkpoint.")
        model_to_use = state.guided_model

    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }

    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    motion_shape = (args.batch_size, model_to_use.njoints, model_to_use.nfeats, n_frames)

    all_motions = []
    all_lengths = []
    all_text = []

    with torch.inference_mode():
        fixseed(args.seed)
        for _ in range(args.num_repetitions):
            sample = state.diffusion.logsnr_sample_loop(
                model_to_use,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
                steps=state.gen_steps,
                use_ddim=False,
            )

            # hml_vec -> XYZ joints
            n_joints = 22
            sample = sample.cpu().permute(0, 2, 3, 1)
            sample = sample * state.std + state.mean
            sample = recover_from_ric(sample.float(), n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_text += model_kwargs["y"]["text"]
            all_motions.append(sample.cpu().numpy())
            _len = model_kwargs["y"]["lengths"].cpu().numpy()
            all_lengths.append(_len)

    total_num_samples = args.num_samples * args.num_repetitions
    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    results = {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
    }

    buf = io.BytesIO()
    np.save(buf, results)
    return buf.getvalue()


app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    app.state.mdm = _init_state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    state = app.state.mdm
    with state.lock:
        payload = _sample(state, req)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=results.npy"},
    )
