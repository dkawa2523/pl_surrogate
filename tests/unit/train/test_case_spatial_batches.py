from __future__ import annotations

import numpy as np

from plasma_surrogate.train.spatial_features import build_case_spatial_features
from plasma_surrogate.train.trainer import train_one_epoch_unet


class _RecordingGridModel:
    output_keys = ["phi"]
    out_channels = 1
    with_rho_eff_head = False
    backend = "numpy"

    def __init__(self, grid_shape: tuple[int, int]):
        self.grid_shape = grid_shape
        self.spatial_batches: list[np.ndarray] = []

    def forward_raw(
        self,
        cond: np.ndarray,
        *,
        training: bool = False,
        spatial_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del cond, training
        if spatial_features is None:
            raise AssertionError("spatial_features must be passed batch-by-batch")
        self.spatial_batches.append(np.asarray(spatial_features, dtype=np.float32).copy())
        bsz = int(spatial_features.shape[0])
        h, w = self.grid_shape
        return np.zeros((bsz, 1, h, w), dtype=np.float32)

    def backward_raw(self, grad_raw: np.ndarray, **kwargs: object) -> dict[str, float]:
        del grad_raw, kwargs
        return {"step_rel_hidden_mean": 0.0, "step_rel_output": 0.0}


def test_train_one_epoch_unet_passes_case_spatial_batches() -> None:
    h, w, c = 4, 5, 3
    model = _RecordingGridModel((h, w))
    cond = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros((2, 1, h, w), dtype=np.float32)
    spatial = np.zeros((2, h, w, c), dtype=np.float32)
    spatial[0, ..., 1] = 1.0
    spatial[1, ..., 1] = 2.0

    train_one_epoch_unet(
        model,
        cond,
        y,
        loss_cfg={"supervised": {"type": "mse", "mask": "all"}},
        spatial_features=spatial,
        batch_size_cases=1,
        shuffle_cases=False,
    )

    assert len(model.spatial_batches) == 2
    assert model.spatial_batches[0].shape == (1, h, w, c)
    assert float(np.mean(model.spatial_batches[0][..., 1])) == 1.0
    assert float(np.mean(model.spatial_batches[1][..., 1])) == 2.0


def test_compact_case_spatial_source_matches_legacy_pack() -> None:
    h, w = 4, 5
    channels = [
        "x",
        "y",
        "mask_plasma",
        "distance_signed",
        "distance_any",
        "mask_coil",
        "distance_coil",
        "coil_proximity",
    ]
    static_data = np.zeros((5, h, w), dtype=np.float32)
    for i in range(5):
        static_data[i] = float(i + 1)
    case_data = np.zeros((2, 3, h, w), dtype=np.float32)
    case_data[0, 0] = 10.0
    case_data[0, 1] = 11.0
    case_data[0, 2] = 12.0
    case_data[1, 0] = 20.0
    case_data[1, 1] = 21.0
    case_data[1, 2] = 22.0

    legacy = np.concatenate(
        [
            np.broadcast_to(static_data.reshape(1, 5, h, w), (2, 5, h, w)),
            case_data,
        ],
        axis=1,
    )
    source, source_name = build_case_spatial_features(
        channels=channels,
        h=h,
        w=w,
        pack={"data": legacy, "channels": np.asarray(channels)},
        static_pack={"data": static_data, "channels": np.asarray(channels[:5])},
        case_pack={"data": case_data, "channels": np.asarray(channels[5:])},
        distance_transform_cfg={"mode": "raw"},
        coord_feature_scaler_artifact={"enabled": False, "channels": {}},
    )

    assert source_name == "compact_case_spatial_pack"
    assert source is not None
    got = source.batch(np.asarray([1, 0], dtype=np.int64))
    expected = np.stack([legacy[1].transpose(1, 2, 0), legacy[0].transpose(1, 2, 0)], axis=0)
    np.testing.assert_allclose(got, expected)
