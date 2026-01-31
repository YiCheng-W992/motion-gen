import torch


class Rotation2xyz:
    """Minimal stub to avoid SMPL dependency.

    This only supports pose_rep='xyz' and returns input as-is.
    """
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset

    def __call__(
        self,
        x,
        mask,
        pose_rep,
        translation,
        glob,
        jointstype,
        vertstrans,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
        **kwargs,
    ):
        if pose_rep != "xyz":
            raise NotImplementedError(
                "rotation2xyz stub only supports pose_rep='xyz'"
            )
        return x
