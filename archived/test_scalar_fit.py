"""Unit tests for scale_estimation.fit_constant_d().

These tests verify that the scalar fit phase correctly recovers the optimal
constant D value for both field and particle modes, as specified in SCALAR_FIT_PLAN.md.

Run with: python test_scalar_fit.py
"""

import numpy as np
import torch

import physics
import varpro
from scale_estimation import estimate_ddi_scale, fit_constant_d
from data import PPPData, sample_ppp_from_field


def test_constant_d_field_mode():
    """Test 1: With known D_true = 0.1 (constant), scalar fit should recover D ≈ 0.1."""
    print("\n" + "="*60)
    print("Test 1: Constant D recovery (field mode)")
    print("="*60)

    # Setup
    d_true = 0.1
    alpha = 0.0
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201

    # Create grid and solve for true u
    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    d_true_vec = d_true * np.ones(n_res)
    u_true_np = physics.fdm_solve_alpha_dirichlet(
        d_true_vec, alpha, mu, x.numpy(), b_true, (source,)
    )
    u_true = torch.tensor(u_true_np, dtype=torch.float64)

    # Run scalar fit with initial guess far from truth
    d_init = 0.05  # 50% underestimate
    d_fit = fit_constant_d(
        x=x,
        alpha=alpha,
        mu=mu,
        sources=(source,),
        u_true=u_true,
        d_init=d_init,
        max_iters=500,
        field_loss="mse",
        verbose=True,
    )

    # Check result
    error = abs(d_fit - d_true)
    rel_error = error / d_true
    print(f"\nResult: d_fit = {d_fit:.6f}, d_true = {d_true:.6f}")
    print(f"Absolute error: {error:.6f}")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.01, f"Scalar fit should recover D within 1%, got {rel_error*100:.2f}%"
    print("✓ PASSED: Constant D recovered within 1%")
    return True


def test_constant_d_particles_mode():
    """Test 2: Scalar fit with particle data should also recover reasonable D."""
    print("\n" + "="*60)
    print("Test 2: Constant D recovery (particles mode)")
    print("="*60)

    # Setup
    d_true = 0.1
    alpha = 0.0
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201
    m_obs = 100  # Number of PPP snapshots

    # Create grid and solve for true u
    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    d_true_vec = d_true * np.ones(n_res)
    u_true_np = physics.fdm_solve_alpha_dirichlet(
        d_true_vec, alpha, mu, x.numpy(), b_true, (source,)
    )
    u_true = torch.tensor(u_true_np, dtype=torch.float64)

    # Sample PPP particles from the field
    rng = np.random.default_rng(42)
    ppp = sample_ppp_from_field(x, u_true, m_obs, rng)
    print(f"Sampled {ppp.n_obs} particles from {m_obs} snapshots")

    # Run scalar fit
    d_init = 0.05
    d_fit = fit_constant_d(
        x=x,
        alpha=alpha,
        mu=mu,
        sources=(source,),
        ppp=ppp,
        d_init=d_init,
        max_iters=500,
        verbose=True,
    )

    # Check result (particles mode has more variance, allow 10% error)
    error = abs(d_fit - d_true)
    rel_error = error / d_true
    print(f"\nResult: d_fit = {d_fit:.6f}, d_true = {d_true:.6f}")
    print(f"Absolute error: {error:.6f}")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.10, f"Scalar fit should recover D within 10%, got {rel_error*100:.2f}%"
    print("✓ PASSED: Constant D recovered within 10%")
    return True


def test_varying_d_recovers_average():
    """Test 3: With varying D_true(x), scalar fit should find a weighted average."""
    print("\n" + "="*60)
    print("Test 3: Varying D recovery (should find weighted average)")
    print("="*60)

    # Setup: sinusoidal D with mean 0.1
    alpha = 0.0
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201

    d_mean = 0.1
    d_amp = 0.04  # D varies from 0.06 to 0.14
    d_freq = 2.0

    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    x_np = x.numpy()
    d_true_np = d_mean + d_amp * np.sin(2 * np.pi * d_freq * x_np)

    u_true_np = physics.fdm_solve_alpha_dirichlet(
        d_true_np, alpha, mu, x_np, b_true, (source,)
    )
    u_true = torch.tensor(u_true_np, dtype=torch.float64)

    # Run scalar fit
    d_init = 0.05
    d_fit = fit_constant_d(
        x=x,
        alpha=alpha,
        mu=mu,
        sources=(source,),
        u_true=u_true,
        d_init=d_init,
        max_iters=500,
        field_loss="mse",
        verbose=True,
    )

    # The optimal constant D should be close to the arithmetic mean
    # (exact relationship depends on the PDE structure)
    d_true_mean = float(np.mean(d_true_np))
    error_from_mean = abs(d_fit - d_true_mean)
    rel_error_from_mean = error_from_mean / d_true_mean

    print(f"\nResult: d_fit = {d_fit:.6f}")
    print(f"Arithmetic mean of D_true: {d_true_mean:.6f}")
    print(f"Error from mean: {error_from_mean:.6f} ({rel_error_from_mean*100:.2f}%)")
    print(f"D_true range: [{np.min(d_true_np):.6f}, {np.max(d_true_np):.6f}]")

    # The fit should be within 20% of the mean (weighted average may differ)
    assert rel_error_from_mean < 0.20, f"Scalar fit should be within 20% of mean, got {rel_error_from_mean*100:.2f}%"
    print("✓ PASSED: Scalar fit found reasonable average")
    return True


def test_scalar_fit_vs_ddi():
    """Test 4: Compare DDI estimate vs scalar fit accuracy."""
    print("\n" + "="*60)
    print("Test 4: DDI vs Scalar Fit comparison")
    print("="*60)

    # Setup
    d_true = 0.1
    alpha = 0.0
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201

    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    d_true_vec = d_true * np.ones(n_res)
    u_true_np = physics.fdm_solve_alpha_dirichlet(
        d_true_vec, alpha, mu, x.numpy(), b_true, (source,)
    )
    u_true = torch.tensor(u_true_np, dtype=torch.float64)

    # Get DDI estimate
    d_ddi = estimate_ddi_scale(
        mu=mu,
        z=source,
        u_field=u_true,
        x_grid=x,
    )

    # Run scalar fit starting from DDI
    d_scalar = fit_constant_d(
        x=x,
        alpha=alpha,
        mu=mu,
        sources=(source,),
        u_true=u_true,
        d_init=d_ddi,
        max_iters=500,
        field_loss="mse",
        verbose=True,
    )

    # Compare
    ddi_error = abs(d_ddi - d_true) / d_true
    scalar_error = abs(d_scalar - d_true) / d_true

    print(f"\nResults:")
    print(f"  D_true:        {d_true:.6f}")
    print(f"  DDI estimate:  {d_ddi:.6f} (error: {ddi_error*100:.2f}%)")
    print(f"  Scalar fit:    {d_scalar:.6f} (error: {scalar_error*100:.2f}%)")
    print(f"  Improvement:   {(ddi_error - scalar_error)/ddi_error*100:.1f}%")

    # Scalar fit should be at least as good as DDI
    assert scalar_error <= ddi_error + 0.01, "Scalar fit should be at least as accurate as DDI"
    print("✓ PASSED: Scalar fit is at least as accurate as DDI")
    return True


def test_rle_loss_mode():
    """Test 5: Scalar fit with RLE (relative log error) loss."""
    print("\n" + "="*60)
    print("Test 5: RLE loss mode")
    print("="*60)

    # Setup
    d_true = 0.1
    alpha = 0.0
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201

    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    d_true_vec = d_true * np.ones(n_res)
    u_true_np = physics.fdm_solve_alpha_dirichlet(
        d_true_vec, alpha, mu, x.numpy(), b_true, (source,)
    )
    u_true = torch.tensor(u_true_np, dtype=torch.float64)

    # Run scalar fit with RLE loss
    d_fit = fit_constant_d(
        x=x,
        alpha=alpha,
        mu=mu,
        sources=(source,),
        u_true=u_true,
        d_init=0.05,
        max_iters=500,
        field_loss="rle",
        verbose=True,
    )

    rel_error = abs(d_fit - d_true) / d_true
    print(f"\nResult: d_fit = {d_fit:.6f}, d_true = {d_true:.6f}")
    print(f"Relative error: {rel_error*100:.2f}%")

    assert rel_error < 0.02, f"RLE mode should recover D within 2%, got {rel_error*100:.2f}%"
    print("✓ PASSED: RLE loss mode works correctly")
    return True


def test_different_alpha_values():
    """Test 6: Scalar fit with different alpha (stochastic convention) values."""
    print("\n" + "="*60)
    print("Test 6: Different alpha values")
    print("="*60)

    d_true = 0.1
    mu = 1.0
    b_true = 100.0
    source = 0.5
    n_res = 201

    x = torch.linspace(0.0, 1.0, n_res, dtype=torch.float64)
    d_true_vec = d_true * np.ones(n_res)

    for alpha in [0.0, 0.5, 1.0]:
        print(f"\n  Testing alpha = {alpha}:")
        u_true_np = physics.fdm_solve_alpha_dirichlet(
            d_true_vec, alpha, mu, x.numpy(), b_true, (source,)
        )
        u_true = torch.tensor(u_true_np, dtype=torch.float64)

        d_fit = fit_constant_d(
            x=x,
            alpha=alpha,
            mu=mu,
            sources=(source,),
            u_true=u_true,
            d_init=0.05,
            max_iters=300,
            field_loss="mse",
            verbose=False,
        )

        rel_error = abs(d_fit - d_true) / d_true
        print(f"    d_fit = {d_fit:.6f}, error = {rel_error*100:.2f}%")
        assert rel_error < 0.02, f"Failed for alpha={alpha}: error = {rel_error*100:.2f}%"

    print("\n✓ PASSED: All alpha values work correctly")
    return True


if __name__ == "__main__":
    print("="*60)
    print("SCALAR FIT UNIT TESTS")
    print("="*60)

    tests = [
        test_constant_d_field_mode,
        test_constant_d_particles_mode,
        test_varying_d_recovers_average,
        test_scalar_fit_vs_ddi,
        test_rle_loss_mode,
        test_different_alpha_values,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed, {failed} failed")
    print("="*60)
