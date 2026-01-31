#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch

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


class _DummyData:
    dataset = type("Dataset", (), {})()


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HumanML text inference (trans_dec).")
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    parser.add_argument(
        "--out",
        default="outputs/sample.npy",
        type=str,
        help="Output .npy path for results dict.",
    )
    parser.add_argument(
        "--sample_steps",
        default=0,
        type=int,
        help="Sampling steps for logSNR spacing. 0 = use diffusion_steps.",
    )
    parser.add_argument(
        "--force_use_ema",
        action="store_true",
        help="Force EMA weights if present in the checkpoint.",
    )
    parser.add_argument(
        "--mp4_dir",
        default="",
        type=str,
        help="Directory to save MP4 files (set to enable rendering).",
    )
    args = parser.parse_args()

    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args = load_args_from_model(args, args_to_overwrite)

    if args.force_use_ema:
        args.use_ema = True
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return apply_rules(args)


def _resolve_data_root(args: argparse.Namespace) -> str:
    if getattr(args, "data_dir", ""):
        return args.data_dir
    # HumanML default
    return "./dataset/HumanML3D"


def main():
    args = _build_args()
    dist_util.setup_dist(args.device)

    cond_mode = get_cond_mode(args)
    if cond_mode != "text":
        raise SystemExit(f"Only text-conditioned models are supported, got {cond_mode!r}.")
    if args.arch != "trans_dec":
        raise SystemExit("Only trans_dec checkpoints are supported in this minimal script.")
    if args.text_encoder_type != "bert":
        raise SystemExit("Only BERT text encoder is supported in this minimal script.")
    if not args.text_prompt and not args.input_text:
        raise SystemExit("Provide --text_prompt or --input_text.")

    max_frames = 196
    fps = 20
    n_frames = min(max_frames, int(args.motion_length * fps))
    if n_frames <= 0:
        raise SystemExit("motion_length too small for current FPS.")

    if args.text_prompt:
        texts = [args.text_prompt] * args.num_samples
    else:
        if not os.path.exists(args.input_text):
            raise SystemExit(f"input_text not found: {args.input_text}")
        with open(args.input_text, "r") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        args.num_samples = len(texts)

    args.batch_size = args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, _DummyData())

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }

    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    sample_steps = args.sample_steps or args.diffusion_steps
    motion_shape = (args.batch_size, model.njoints, model.nfeats, n_frames)

    all_motions = []
    all_lengths = []
    all_text = []

    data_root = _resolve_data_root(args)
    mean = np.load(os.path.join(data_root, "Mean.npy"))
    std = np.load(os.path.join(data_root, "Std.npy"))

    with torch.inference_mode():
        fixseed(args.seed)
        for rep_i in range(args.num_repetitions):
            print(f"Sampling repetition {rep_i + 1}/{args.num_repetitions}...")
            sample = diffusion.logsnr_sample_loop(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
                steps=sample_steps,
                use_ddim=False,
            )

            # hml_vec -> XYZ joints
            n_joints = 22
            sample = sample.cpu().permute(0, 2, 3, 1)
            sample = sample * std + mean
            sample = recover_from_ric(sample.float(), n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_motions.append(sample.cpu().numpy())
            _len = model_kwargs["y"]["lengths"].cpu().numpy()
            all_lengths.append(_len)
            all_text += model_kwargs["y"]["text"]

    total_num_samples = args.num_samples * args.num_repetitions
    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_text = all_text[:total_num_samples]

    results = {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "fps": fps,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "model_path": args.model_path,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, results)
    print(f"Saved results to {args.out}")

    mp4_dir = args.mp4_dir.strip()
    if mp4_dir:
        from data_loaders.humanml.utils.plot_script import plot_3d_motion
        from data_loaders.humanml.utils import paramUtil

        os.makedirs(mp4_dir, exist_ok=True)

        skeleton = paramUtil.kit_kinematic_chain if args.dataset == "kit" else paramUtil.t2m_kinematic_chain
        total_num_samples = args.num_samples * args.num_repetitions
        print(f"Rendering {total_num_samples} MP4 files to {mp4_dir}...")

        for rep_i in range(args.num_repetitions):
            for sample_i in range(args.num_samples):
                idx = rep_i * args.num_samples + sample_i
                motion = all_motions[idx].transpose(2, 0, 1)
                length = int(all_lengths[idx])
                motion = motion[:length]

                caption = all_text[idx]
                save_file = f"sample{sample_i:02d}_rep{rep_i:02d}.mp4"
                save_path = os.path.join(mp4_dir, save_file)

                clip = plot_3d_motion(
                    save_path,
                    skeleton,
                    motion,
                    title=caption,
                    dataset=args.dataset,
                    fps=fps,
                )
                clip.write_videofile(save_path, fps=fps, threads=4, logger=None)
                clip.close()
        print("MP4 rendering complete.")


if __name__ == "__main__":
    main()
