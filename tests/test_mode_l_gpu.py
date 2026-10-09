"""Mode L on the GPU filters the rotated target with the Laplacian, as on the CPU."""

import pytest
import torch

from vesper.data.map import EMmap, unify_dims
from vesper.fitter import MapFitter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _fit(inputs, mode, gpu):
    ref, tgt = EMmap(str(inputs / "a.mrc")), EMmap(str(inputs / "target.mrc"))
    for em_map in (ref, tgt):
        em_map.set_vox_size(thr=0.05, voxel_size=3.0)
    unify_dims([ref, tgt], voxel_size=3.0)
    for em_map in (ref, tgt):
        em_map.resample_and_vec(dreso=8.0)
    device = torch.device("cuda:0") if gpu else None
    fitter = MapFitter(
        ref, tgt, 30.0, mode, True, None, None, None, 2, gpu, device, topn=5
    )
    fitter.fit()
    return fitter.final_list


@pytest.mark.parametrize("mode", ["L", "C"])
def _key(items):
    return [(tuple(i["angle"]), tuple(int(v) for v in i["vox_trans"])) for i in items]


@pytest.mark.parametrize("mode", ["L", "C"])
def test_gpu_and_cpu_give_the_same_top_poses(inputs, mode):
    cpu, gpu = _fit(inputs, mode, False), _fit(inputs, mode, True)
    assert _key(gpu) == _key(cpu)
    assert [i["score"] for i in gpu] == pytest.approx(
        [i["score"] for i in cpu], rel=1e-4
    )
