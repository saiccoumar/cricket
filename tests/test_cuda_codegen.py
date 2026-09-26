"""Checks the `language="cuda"` backend against pinocchio.

The generated header is compiled for the host with cricket's own JIT (it is `__host__ __device__`), and its
pose, Jacobian and sphere outputs are compared with pinocchio at random configurations. When `nvcc` and a GPU
are available, the same header is also compiled for the device and compared with the host build.

    python -m pytest tests/test_cuda_codegen.py -s
"""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

import cricket
from cricket import _core_ext

ROOT = Path(__file__).resolve().parents[1] / "resources"
ROBOTS = ["panda", "fr3", "ur5", "baxter", "fetch", "pr2"]
N_SAMPLES = 1000
HOST_FN = ctypes.CFUNCTYPE(None, *[ctypes.POINTER(ctypes.c_float)] * 4, ctypes.c_int)

WRAPPER = r"""
#include "robot.cuh"
namespace r = cricket::robots::{ns};
extern "C" void batch_{ns}(const float *q, float *pose, float *jac, float *spheres, int n)
{{
    for (int i = 0; i < n; ++i)
    {{
        r::ee_pose_jacobian(q + i * r::n_q, pose + i * 7, jac + i * 6 * r::n_q);
        r::sphere_centers(q + i * r::n_q, spheres + i * 4 * r::n_spheres);
    }}
}}
"""

DEVICE_WRAPPER = r"""
#include "robot.cuh"
namespace r = cricket::robots::{ns};
__global__ void kernel(const float *q, float *pose, float *jac, float *spheres, int n)
{{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    r::ee_pose_jacobian(q + i * r::n_q, pose + i * 7, jac + i * 6 * r::n_q);
    r::sphere_centers(q + i * r::n_q, spheres + i * 4 * r::n_spheres);
}}
extern "C" int batch(const float *q, float *pose, float *jac, float *spheres, int n)
{{
    const size_t sizes[4] = {{sizeof(float) * n * r::n_q, sizeof(float) * n * 7,
                             sizeof(float) * n * 6 * r::n_q, sizeof(float) * n * 4 * r::n_spheres}};
    float *d[4];
    for (int k = 0; k < 4; ++k) cudaMalloc(&d[k], sizes[k]);
    cudaMemcpy(d[0], q, sizes[0], cudaMemcpyHostToDevice);
    kernel<<<(n + 127) / 128, 128>>>(d[0], d[1], d[2], d[3], n);
    cudaMemcpy(pose, d[1], sizes[1], cudaMemcpyDeviceToHost);
    cudaMemcpy(jac, d[2], sizes[2], cudaMemcpyDeviceToHost);
    cudaMemcpy(spheres, d[3], sizes[3], cudaMemcpyDeviceToHost);
    for (int k = 0; k < 4; ++k) cudaFree(d[k]);
    return (int)cudaGetLastError();
}}
"""


def generate(robot: str, frames: list[str] | None = None):
    cfg = json.loads((ROOT / f"{robot}.json").read_text())
    data = {"name": cfg["name"]}
    if frames is not None:
        data["trace_frames"] = frames
    opts = cricket.GenOptions(
        urdf=ROOT / cfg["urdf"],
        srdf=ROOT / cfg["srdf"],
        end_effector=cfg["end_effector"],
        language="cuda",
        data=data,
    )
    return cfg, cricket.generate_robot_source(opts)


def call_batch(fn, q: np.ndarray, n_spheres: int):
    n, n_q = q.shape
    q = np.ascontiguousarray(q, dtype=np.float32)
    pose = np.empty((n, 7), np.float32)
    jac = np.empty((n, 6, n_q), np.float32)
    spheres = np.empty((n, n_spheres, 4), np.float32)
    ptr = ctypes.POINTER(ctypes.c_float)
    rc = fn(*(a.ctypes.data_as(ptr) for a in (q, pose, jac, spheres)), ctypes.c_int(n))
    assert not rc, f"batch returned CUDA error {rc}"
    return pose, jac, spheres


@pytest.fixture(scope="module")
def jit():
    return _core_ext.jit.JitSession()


@pytest.mark.parametrize("robot", ROBOTS)
def test_matches_pinocchio(robot, jit, tmp_path):
    cfg, gen = generate(robot)
    ns = cfg["name"].lower()
    (tmp_path / "robot.cuh").write_text(gen.source)

    opts = _core_ext.jit.CompileOptions()
    opts.include_dirs = [str(tmp_path)]
    jit.add_source(WRAPPER.format(ns=ns), opts)
    fn = HOST_FN(jit.lookup_address(f"batch_{ns}"))

    model = pin.buildModelFromUrdf(str(ROOT / cfg["urdf"]), mimic=True)  # as RobotInfo builds it
    data = model.createData()
    fid = model.getFrameId(cfg["end_effector"])
    geom = pin.buildGeomFromUrdf(model, str(ROOT / cfg["urdf"]), pin.GeometryType.COLLISION)

    rng = np.random.default_rng(0)
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit
    q = rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi), size=(N_SAMPLES, model.nq))

    pose, jac, spheres = call_batch(fn, q, gen.n_spheres)

    for i in range(N_SAMPLES):
        pin.computeJointJacobians(model, data, q[i])
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[fid]
        J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)

        np.testing.assert_allclose(pose[i, 4:], oMf.translation, atol=1e-4)
        quat = pin.Quaternion(oMf.rotation)
        ref = np.array([quat.w, quat.x, quat.y, quat.z])
        assert min(np.abs(pose[i, :4] - ref).max(), np.abs(pose[i, :4] + ref).max()) < 1e-4
        np.testing.assert_allclose(jac[i], J, atol=1e-4)

        gdata = pin.GeometryData(geom)
        pin.updateGeometryPlacements(model, data, geom, gdata)
        centers = np.array([gdata.oMg[k].translation for k in range(geom.ngeoms)])
        np.testing.assert_allclose(spheres[i, :, :3], centers, atol=1e-4)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available")
@pytest.mark.parametrize("robot", ["panda", "baxter"])
def test_device_matches_host(robot, jit, tmp_path):
    cfg, gen = generate(robot)
    ns = cfg["name"].lower()
    (tmp_path / "robot.cuh").write_text(gen.source)

    (tmp_path / "host.cc").write_text(WRAPPER.format(ns=ns))
    (tmp_path / "device.cu").write_text(DEVICE_WRAPPER.format(ns=ns))
    flags = ["-O2", "-shared", "-o"]
    subprocess.run(["c++", "-fPIC", *flags, str(tmp_path / "host.so"), str(tmp_path / "host.cc")], check=True)
    subprocess.run(
        ["nvcc", "-arch=native", "-Xcompiler", "-fPIC", *flags, str(tmp_path / "device.so"), str(tmp_path / "device.cu")],
        check=True)

    host = getattr(ctypes.CDLL(str(tmp_path / "host.so")), f"batch_{ns}")
    host.restype = None
    device = ctypes.CDLL(str(tmp_path / "device.so")).batch
    device.restype = ctypes.c_int

    q = np.random.default_rng(1).uniform(-1.0, 1.0, size=(4096, gen.dimension))
    h = call_batch(host, q, gen.n_spheres)
    d = call_batch(device, q, gen.n_spheres)
    for a, b in zip(h, d):
        np.testing.assert_allclose(a, b, atol=1e-5)


FRAMES_WRAPPER = r"""
#include "robot.cuh"
namespace r = cricket::robots::{ns};
extern "C" void frames_{ns}(const float *q, float *poses, int n)
{{
    for (int i = 0; i < n; ++i) r::frame_poses(q + i * r::n_q, poses + i * 7 * r::n_frames);
}}
"""


@pytest.mark.parametrize("robot", ["panda", "baxter"])
def test_frame_poses_match_pinocchio(robot, jit, tmp_path):
    cfg = json.loads((ROOT / f"{robot}.json").read_text())
    model = pin.buildModelFromUrdf(str(ROOT / cfg["urdf"]), mimic=True)  # as RobotInfo builds it
    data = model.createData()
    kinds = (pin.FrameType.JOINT, pin.FrameType.FIXED_JOINT)
    frame_ids = [i for i, f in enumerate(model.frames) if f.type in kinds]
    frames = [model.frames[i].name for i in frame_ids]

    _, gen = generate(robot, frames)
    ns = cfg["name"].lower()
    (tmp_path / "robot.cuh").write_text(gen.source)
    opts = _core_ext.jit.CompileOptions()
    opts.include_dirs = [str(tmp_path)]
    jit.add_source(FRAMES_WRAPPER.format(ns=ns), opts)
    fn = ctypes.CFUNCTYPE(None, *[ctypes.POINTER(ctypes.c_float)] * 2, ctypes.c_int)(
        jit.lookup_address(f"frames_{ns}"))

    q = np.random.default_rng(2).uniform(-1.0, 1.0, size=(200, model.nq)).astype(np.float32)
    poses = np.empty((len(q), len(frames), 7), np.float32)
    ptr = ctypes.POINTER(ctypes.c_float)
    fn(q.ctypes.data_as(ptr), poses.ctypes.data_as(ptr), len(q))

    for i in range(len(q)):
        pin.framesForwardKinematics(model, data, q[i].astype(np.float64))
        for k in range(len(frames)):
            oMf = data.oMf[frame_ids[k]]
            np.testing.assert_allclose(poses[i, k, 4:], oMf.translation, atol=1e-4)
            quat = pin.Quaternion(oMf.rotation)
            ref = np.array([quat.w, quat.x, quat.y, quat.z])
            assert min(np.abs(poses[i, k, :4] - ref).max(), np.abs(poses[i, k, :4] + ref).max()) < 1e-4


def test_constant_pose_compiles(jit, tmp_path):
    """An end-effector fixed to the root traces to a constant pose with no temporaries."""
    cfg = json.loads((ROOT / "panda.json").read_text())
    gen = cricket.generate_robot_source(cricket.GenOptions(
        urdf=ROOT / cfg["urdf"], srdf=ROOT / cfg["srdf"], end_effector="panda_link0",
        language="cuda", data={"name": "PandaRoot"}))
    (tmp_path / "robot.cuh").write_text(gen.source)
    opts = _core_ext.jit.CompileOptions()
    opts.include_dirs = [str(tmp_path)]
    jit.add_source(WRAPPER.format(ns="pandaroot"), opts)
    q = np.zeros((1, gen.dimension))
    pose, jac, _ = call_batch(HOST_FN(jit.lookup_address("batch_pandaroot")), q, gen.n_spheres)
    np.testing.assert_allclose(pose[0], [1, 0, 0, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(jac, 0, atol=1e-6)
