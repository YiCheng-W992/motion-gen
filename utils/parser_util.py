from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    if args.model_path != "":
        args = load_args_from_model(args, args_to_overwrite)
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return apply_rules(args)


def load_args_from_model(args, args_to_overwrite):
    model_path = args.model_path
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])
    return args


def apply_rules(args):
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError("group_name was not found.")


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument("--model_path")
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except Exception as exc:
        raise ValueError("model_path argument must be specified.") from exc


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--local_rank", default=-1, type=int, help="DDP local rank.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    group.add_argument("--diffusion_steps", default=1000, type=int)
    group.add_argument("--sigma_small", default=True, type=bool)


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--arch", default="trans_dec", choices=["trans_dec"], type=str)
    group.add_argument("--text_encoder_type", default="bert", choices=["bert"], type=str)
    group.add_argument("--emb_trans_dec", action="store_true")
    group.add_argument("--layers", default=8, type=int)
    group.add_argument("--model_dim", default=512, type=int)
    group.add_argument("--cond_mask_prob", default=0.1, type=float)
    group.add_argument("--mask_frames", action="store_true")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float)
    group.add_argument("--lambda_vel", default=0.0, type=float)
    group.add_argument("--lambda_fc", default=0.0, type=float)
    group.add_argument("--lambda_target_loc", default=0.0, type=float)
    group.add_argument("--pos_embed_max_len", default=5000, type=int)
    group.add_argument("--use_ema", action="store_true")


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset", default="humanml", choices=["humanml"], type=str)
    group.add_argument("--data_dir", default="", type=str)


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--output_dir", default="", type=str)
    group.add_argument("--num_samples", default=6, type=int)
    group.add_argument("--num_repetitions", default=3, type=int)
    group.add_argument("--guidance_param", default=2.5, type=float)


def add_generate_options(parser):
    group = parser.add_argument_group("generate")
    group.add_argument("--motion_length", default=9.8, type=float)
    group.add_argument("--input_text", default="", type=str)
    group.add_argument("--text_prompt", default="", type=str)


def get_cond_mode(args):
    return "text"
