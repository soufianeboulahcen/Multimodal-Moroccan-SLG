"""PyTorch port of the original TensorFlow 1.x backpropagation-based 3D-pose
filtering by Fang et al. (Prompt2Sign / SignLLM, 2024).

The original implementation at
  https://github.com/SignLLM/Prompt2Sign/blob/main/tools/2D_to_3D/pose3D.py
uses TF1 graph-mode (`tf.placeholder`, `tf.Variable`, `sess.run`,
`tf.train.GradientDescentOptimizer`) which cannot run on Python 3.12 / aarch64 /
CUDA 13 (no TF 1.15 wheels exist for that target).

This file replaces the TF1 implementation with a PyTorch eager-mode version that
is algorithmically identical:
  * Same variables (log bone lengths, root positions per frame, limb angles).
  * Same forward pass (skeletal-hierarchy propagation: pos_child = pos_parent +
    L * (Ax, Ay, Az) / |A|).
  * Same loss (weighted reprojection MSE on 2D observed keypoints + L1 reg on
    bone lengths + temporal smoothness in 3D).
  * Same hyperparameters (learningRate=0.1, nCycles=1000,
    regulatorRates=[0.001, 0.1]).
  * Same numerical-stability epsilon (1e-10).

Given identical input arrays, hyperparameters, and seeds, output is numerically
equivalent to the TF1 reference (modulo float32 op-ordering differences).

See docs/DECISIONS.md for the rationale behind the port.
"""
from __future__ import annotations

import numpy
import torch

import skeletalModel


def backpropagationBasedFiltering(
    lines0_values,           # initial (logarithm of) bone lengths, shape (nBones,)
    rootsx0_values,          # head x position per frame, shape (T, 1)
    rootsy0_values,          # head y position per frame, shape (T, 1)
    rootsz0_values,          # head z position per frame, shape (T, 1)
    anglesx0_values,         # x-component of limb angles per frame, shape (T, nLimbs)
    anglesy0_values,
    anglesz0_values,
    tarx_values,             # target 2D x per frame, shape (T, nPoints)
    tary_values,             # target 2D y per frame, shape (T, nPoints)
    w_values,                # OpenPose confidence per frame per point, shape (T, nPoints)
    structure,               # tuple of (parent_idx, child_idx, bone_idx) tuples
    dtype,                   # "float32" — keeps API parity with the TF1 version
    learningRate=0.1,
    nCycles=1000,
    regulatorRates=(0.001, 0.1),
):
    """Refine 3D pose estimate by gradient descent on a kinematic skeleton.

    Returns three numpy arrays (x, y, z), each of shape (T, nPoints) with the
    optimized 3D joint coordinates.
    """
    if dtype != "float32":
        raise ValueError(f"only dtype='float32' is supported (got {dtype!r})")

    torch_dtype = torch.float32
    # Use CUDA if available (the upstream TF1 code did the same implicitly via
    # session device placement). CPU fallback keeps the function portable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = rootsx0_values.shape[0]
    nBones, nPoints = skeletalModel.structureStats(structure)
    nLimbs = len(structure)

    # Learnable variables — initialised from the closed-form Step I/II/III result.
    lines = torch.tensor(lines0_values, dtype=torch_dtype, device=device, requires_grad=True)
    rootsx = torch.tensor(rootsx0_values, dtype=torch_dtype, device=device, requires_grad=True)
    rootsy = torch.tensor(rootsy0_values, dtype=torch_dtype, device=device, requires_grad=True)
    rootsz = torch.tensor(rootsz0_values, dtype=torch_dtype, device=device, requires_grad=True)
    anglesx = torch.tensor(anglesx0_values, dtype=torch_dtype, device=device, requires_grad=True)
    anglesy = torch.tensor(anglesy0_values, dtype=torch_dtype, device=device, requires_grad=True)
    anglesz = torch.tensor(anglesz0_values, dtype=torch_dtype, device=device, requires_grad=True)

    # Targets and weights are observations, not parameters.
    tarx = torch.tensor(tarx_values, dtype=torch_dtype, device=device)
    tary = torch.tensor(tary_values, dtype=torch_dtype, device=device)
    w = torch.tensor(w_values, dtype=torch_dtype, device=device)

    optimizer = torch.optim.SGD(
        [lines, rootsx, rootsy, rootsz, anglesx, anglesy, anglesz],
        lr=learningRate,
    )

    epsilon = 1e-10  # numerical stability for angle normalisation

    for iCycle in range(nCycles):
        optimizer.zero_grad()

        # Forward pass — propagate joint positions along the skeletal tree.
        x_list = [None] * nPoints
        y_list = [None] * nPoints
        z_list = [None] * nPoints
        x_list[0] = rootsx
        y_list[0] = rootsy
        z_list[0] = rootsz

        # `structure` is ordered root-to-leaves; iterating in order guarantees
        # parents are computed before children.
        for i, (a, b, l) in enumerate(structure):
            L = torch.exp(lines[l])
            Ax = anglesx[0:T, i:(i + 1)]
            Ay = anglesy[0:T, i:(i + 1)]
            Az = anglesz[0:T, i:(i + 1)]
            normA = torch.sqrt(Ax * Ax + Ay * Ay + Az * Az) + epsilon
            x_list[b] = x_list[a] + L * Ax / normA
            y_list[b] = y_list[a] + L * Ay / normA
            z_list[b] = z_list[a] + L * Az / normA

        x = torch.cat(x_list, dim=1)  # (T, nPoints)
        y = torch.cat(y_list, dim=1)
        z = torch.cat(z_list, dim=1)

        # Weighted reprojection MSE on 2D (z has no observation; it floats free
        # subject only to the smoothness regularizer).
        loss = (w * (x - tarx) ** 2 + w * (y - tary) ** 2).sum() / (T * nPoints)

        # reg1: L1 penalty on bone lengths (in log space → sum of exp(lines)).
        reg1 = torch.exp(lines).sum()

        # reg2: temporal smoothness on full 3D trajectory.
        dx = x[0:(T - 1), 0:nPoints] - x[1:T, 0:nPoints]
        dy = y[0:(T - 1), 0:nPoints] - y[1:T, 0:nPoints]
        dz = z[0:(T - 1), 0:nPoints] - z[1:T, 0:nPoints]
        reg2 = (dx * dx + dy * dy + dz * dz).sum() / ((T - 1) * nPoints)

        total = loss + regulatorRates[0] * reg1 + regulatorRates[1] * reg2
        total.backward()
        optimizer.step()

        # Print every 100 iterations + the final one. Upstream prints every
        # iteration; with 1000 SGD steps × 2216 clips that's 2.2M lines per
        # full-dataset run, which dominates the log.  This is a verbosity-only
        # change — algorithmic output is unaffected.
        if iCycle % 100 == 0 or iCycle == nCycles - 1:
            print("iCycle = %3d, loss = %e" % (iCycle, loss.item()))

    # Final pose at the optimized parameters (no gradient tracking).
    with torch.no_grad():
        x_list = [None] * nPoints
        y_list = [None] * nPoints
        z_list = [None] * nPoints
        x_list[0] = rootsx
        y_list[0] = rootsy
        z_list[0] = rootsz
        for i, (a, b, l) in enumerate(structure):
            L = torch.exp(lines[l])
            Ax = anglesx[0:T, i:(i + 1)]
            Ay = anglesy[0:T, i:(i + 1)]
            Az = anglesz[0:T, i:(i + 1)]
            normA = torch.sqrt(Ax * Ax + Ay * Ay + Az * Az) + epsilon
            x_list[b] = x_list[a] + L * Ax / normA
            y_list[b] = y_list[a] + L * Ay / normA
            z_list[b] = z_list[a] + L * Az / normA
        x = torch.cat(x_list, dim=1).cpu().numpy()
        y = torch.cat(y_list, dim=1).cpu().numpy()
        z = torch.cat(z_list, dim=1).cpu().numpy()

    return [x, y, z]


if __name__ == "__main__":
    # Sanity-check on the trivial three-bone tree the original file used.
    structure = (
        (0, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
    )

    T = 3
    nBones, nPoints = skeletalModel.structureStats(structure)
    nLimbs = len(structure)

    lines0_values = numpy.zeros((nBones,), dtype="float32")
    rootsx0_values = numpy.ones((T, 1), dtype="float32")
    rootsy0_values = numpy.ones((T, 1), dtype="float32")
    rootsz0_values = numpy.ones((T, 1), dtype="float32")
    anglesx0_values = numpy.ones((T, nLimbs), dtype="float32")
    anglesy0_values = numpy.ones((T, nLimbs), dtype="float32")
    anglesz0_values = numpy.ones((T, nLimbs), dtype="float32")
    w_values = numpy.ones((T, nPoints), dtype="float32")
    tarx_values = numpy.ones((T, nPoints), dtype="float32")
    tary_values = numpy.ones((T, nPoints), dtype="float32")

    x, y, z = backpropagationBasedFiltering(
        lines0_values,
        rootsx0_values, rootsy0_values, rootsz0_values,
        anglesx0_values, anglesy0_values, anglesz0_values,
        tarx_values, tary_values, w_values,
        structure, "float32",
        nCycles=10,  # short for the smoke test
    )
    print(f"smoke test ok: x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")
