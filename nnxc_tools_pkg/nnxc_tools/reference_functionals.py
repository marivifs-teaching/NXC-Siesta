import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from pyscf import gto, dft

from typing import Union
import jax.typing as jt
import numpy.typing as npt

_PBE_KAPPA = 0.804
_PBE_MU    = 0.2195149727645171
_PBE_BETA  = 0.06672455060314922

# TODO DOC - see where the documentation is needed, and put it.


# ---------OBTAIN THE VALUES WE NEED ----------------
def get_true_values_FX_grid(npts: int,
                            s_lims: list[float], rho_lims: list[float],
                            separe_s: bool = False,
                            separe_s_val: float = 0.7, lim_s: float = 1.0,
                            erho: bool = False,
                            return_rhospoints: bool = False) \
                            -> tuple:
    """Get the true values and the inputs for Fx.

    inputs = rho, grad_rho
    Being Fx the exchange enchancement factor. It retuns the values of Fx
    and the gradients (w.r.t. the density and the derivative of the density)
    in an uniform grid, that is defined by the inputs of the functions.
    Note that these inputs are also returned.

    Args:
        npts (int): total number of points in the grid. Therefore, rho and
            s will have sqrt(npts) points each.
        s_lims (list[float]): limits of the s values.
        rho_lims (list[float]): limits of the rho values.
        separe_s (bool, optional): if True, the s values will be separated in
            two groups, one from s_lims[0] to lim_s, and the other from lim_s
            to s_lims[1]. Defaults to False.
        separe_s_val (float, optional): value percentage of the s values that
            will be in the first group. This means we will have separe_s_val *
            sqrt(npts) points in the first group, and the rest in the second
            group. Only used if separe_s=True. Defaults to 0.7.
        lim_s (float, optional): value of s that separates the two groups of
            s values. Only used if separe_s=True. Defaults to 1.0.
        erho (bool, optional): Wheher to derivate e*rho = e_{UEG}*Fx*rho.
            Defaults to False.

    Returns:
        tuple:
            - inputs (jt.ArrayLike[npts, 2]): inputs used to evaluate Fx
                (rho, grad_rho).
            - true_fx (jt.ArrayLike[npts]): true values of Fx evaluated in
                "inputs".
            - true_fx_grad (jt.ArrayLike[npts, 2]): gradients of Fx w.r.t. rho
                and grad_rho evaluated in "inputs".
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    # Create input array
    inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)

    # Calculate true Fx
    true_fx = Fx(rho_flat, grad_rho_flat)
    if erho:
        def Fx_erho(rho, grho):
            e_x_ueg = lda_x(rho)
            return Fx(rho, grho) * rho * e_x_ueg
        true_fx_grad = jax.vmap(
            jax.grad(Fx_erho, argnums=(0, 1)))(rho_flat, grad_rho_flat)
    else:

        def scalar_Fx(rho, grad_rho):
            return Fx(jnp.array([rho]), jnp.array([grad_rho]))[0]

        # Compute gradients
        grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
        true_fx_grad = grad_Fx(rho_flat, grad_rho_flat)
    if not return_rhospoints:
        return inputs, true_fx, true_fx_grad
    else:
        return inputs, true_fx, true_fx_grad, rho_values, s_values


def get_true_values_FC_grid(npts: int,
                            s_lims: list[float], rho_lims: list[float],
                            separe_s: bool = False,
                            separe_s_val: float = 0.7, lim_s: float = 1.0,
                            erho: bool = False, return_rhospoints: bool = False) \
                            -> tuple:
    """Get the true values and the inputs for Fc.

    inputs = rho, grad_rho.

    Being Fc the correlation enchancement factor. It retuns the values of Fc
    and the gradients (w.r.t. the density and the derivative of the density)
    in an uniform grid, that is defined by the inputs of the functions.
    Note that these inputs are also returned.

    Args:
        npts (int): total number of points in the grid. Therefore, rho and
            s will have sqrt(npts) points each.
        s_lims (list[float]): limits of the s values.
        rho_lims (list[float]): limits of the rho values.
        separe_s (bool, optional): if True, the s values will be separated in
            two groups, one from s_lims[0] to lim_s, and the other from lim_s
            to s_lims[1]. Defaults to False.
        separe_s_val (float, optional): value percentage of the s values that
            will be in the first group. This means we will have separe_s_val *
            sqrt(npts) points in the first group, and the rest in the second
            group. Only used if separe_s=True. Defaults to 0.7.
        lim_s (float, optional): value of s that separates the two groups of
            s values. Only used if separe_s=True. Defaults to 1.0.
        erho  (bool, optional): Wheher to derivate e*rho = e_{UEG}*Fx*rho.
            Defaults to False.

    Returns:
        tuple:
            - inputs (jt.ArrayLike[npts, 2]): inputs used to evaluate Fc
                (rho, grad_rho).
            - true_fc (jt.ArrayLike[npts]): true values of Fc evaluated in
                "inputs".
            - true_fc_grad (jt.ArrayLike[npts, 2]): gradients of Fc w.r.t. rho
                and grad_rho evaluated in "inputs".
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    # Create input array
    inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)

    # Calculate true Fc values
    true_fc = Fc(rho_flat, grad_rho_flat)

    if erho:
        def Fc_erho(rho, grho):
            e_x_ueg = pw92c_unpolarized_scalar(rho)
            return Fc_scalar(rho, grho) * rho * e_x_ueg
        true_fc_grad = jax.vmap(
            jax.grad(Fc_erho, argnums=(0, 1)))(rho_flat, grad_rho_flat)
    else:
        def scalar_Fc(rho, grad_rho):
            return Fc(jnp.array([rho]), jnp.array([grad_rho]))[0]

        # Compute gradients
        grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
        true_fc_grad = grad_Fc(rho_flat, grad_rho_flat)
    if not return_rhospoints:
        return inputs, true_fc, true_fc_grad
    else:
        return inputs, true_fc, true_fc_grad, rho_values, s_values


def get_true_values_FXC_grid(npts: int,
                             s_lims: list[float], rho_lims: list[float],
                             separe_s: bool = False,
                             separe_s_val: float = 0.7, lim_s: float = 1.0,
                             Fx_ref=None, Fc_ref=None) \
                             -> tuple:
    """Get the true values and the inputs for Fx and Fc.

    inputs = rho, grad_rho.
    Being Fx and Fc the exchange and correlation enchancement factors,
    respectively. It retuns the values of Fx and Fc, and the gradients
    (w.r.t. the density and the derivative of the density), for both cases,
    in the defined grid, that is defined by the inputs of the functions.
    Note that these inputs are also returned ([rho, grad_rho]).

    Args:
        npts (int): total number of points in the grid. Therefore, rho and
            s will have sqrt(npts) points each.
        s_lims (list[float]): limits of the s values.
        rho_lims (list[float]): limits of the rho values.
        separe_s (bool, optional): if True, the s values will be separated in
            two groups, one from s_lims[0] to lim_s, and the other from lim_s
            to s_lims[1]. Defaults to False.
        separe_s_val (float, optional): value percentage of the s values that
            will be in the first group. This means we will have separe_s_val *
            sqrt(npts) points in the first group, and the rest in the second
            group. Only used if separe_s=True. Defaults to 0.7.
        lim_s (float, optional): value of s that separates the two groups of
            s values. Only used if separe_s=True. Defaults to 1.0.
        Fx_ref (Union[Callable, None], optional): reference function for Fx.
            If None, the default Fx function will be used. Defaults to None.
        Fc_ref (Union[Callable, None], optional): reference function for Fc.
            If None, the default Fc function will be used. Defaults to None.

    Returns:
        tuple:
            - inputs: inputs used to evaluate Fx and Fc ([rho, grad_rho]).
            - true_fx: true values of Fx evaluated in "inputs".
            - true_fx_grad: gradients of Fx w.r.t. rho and grad_rho evaluated
                in "inputs".
            - true_fc: true values of Fc evaluated in "inputs".
            - true_fc_grad: gradients of Fc w.r.t. rho and grad_rho evaluated
                in "inputs".
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s
    if Fx_ref is None:
        Fx_ref = Fx
    if Fc_ref is None:
        Fc_ref = Fc

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    # Create input array
    inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)

    # Calculate true Fx and Fc
    true_fx = Fx_ref(rho_flat, grad_rho_flat)
    true_fc = Fc_ref(rho_flat, grad_rho_flat)

    def scalar_Fx(rho, grad_rho):
        return Fx_ref(jnp.array([rho]), jnp.array([grad_rho]))[0]

    def scalar_Fc(rho, grad_rho):
        return Fc_ref(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
    true_fx_grad = grad_Fx(rho_flat, grad_rho_flat)
    grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
    true_fc_grad = grad_Fc(rho_flat, grad_rho_flat)

    return inputs, true_fx, true_fx_grad, true_fc, true_fc_grad


def gtv_FX_grid_vgrho_dvsig(npts: int, s_lims: list[float], rho_lims: list[float],
                            separe_s: bool = False,
                            separe_s_val: float = 0.7, lim_s: float = 1.0,
                            erho=True, return_sigma=False) \
                            -> tuple:
    """Get the true values, gradients and the inputs for Fx.

    inputs:
        - If return_sigma is false: inputs = rho, grad_rho.
        - If return_sigma is true: inputs = rho, sigma.
    !! BUT the gradients are computed wrt rho and sigma!!
    TODO: DOC
    Args:
        erho (bool, optional): Wheher to derivate e*rho = e_{UEG}*Fx*rho.
        Defaults to True.
    Returns:
        tuple: grho_inputs, true_fx, true_fx_gradsig
            - grho_inputs: inputs for the model. rho, grad_rho = grho_inputs
            - true_fx: true values of Fx. Fx(rho, grad_rho)
            - true_fx_gradsig: true gradients of Fx. dFx/drho, dFx/dsigma
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()
    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grho_flat = (2 * s_flat * k_F * rho_flat)
    sigma_flat = grho_flat**2

    # Create input array
    grho_inputs = jnp.stack([rho_flat, grho_flat], axis=1)
    sig_inputs = jnp.stack([rho_flat, sigma_flat], axis=1)

    # Calculate true Fx
    true_fx = Fx(rho_flat, grho_flat)
    if erho:
        def FxSIG_erho(rho, sigma):
            e_x_ueg = lda_x(rho)
            return FxSIG(rho, sigma) * rho * e_x_ueg
        true_fx_gradsig = jax.vmap(
            jax.grad(FxSIG_erho, argnums=(0, 1)))(rho_flat, sigma_flat)
    else:
        true_fx_gradsig = jax.vmap(
            jax.grad(FxSIG, argnums=(0, 1)))(rho_flat, sigma_flat)
    if return_sigma:
        return sig_inputs, true_fx, true_fx_gradsig
    else:
        return grho_inputs, true_fx, true_fx_gradsig


def gtv_FC_grid_vgrho_dvsig(npts: int, s_lims: list[float],
                            rho_lims: list[float],
                            separe_s: bool = False,
                            separe_s_val: float = 0.7, lim_s: float = 1.0,
                            erho=True, return_sigma=False) \
                            -> tuple:
    """Get the true values, gradients and the inputs for Fc.

    The gradients are computed wrt rho and sigma.
    rho, sigma = inputs
    TODO: DOC
    Args:
        erho (bool, optional): Wheher to derivate e*rho = e_{UEG}*Fc*rho.
            Defaults to True.
    Returns:
        tuple: grho_inputs, true_fc, true_fc_gradsig
            - grho_inputs: inputs for the model. rho, grad_rho = grho_inputs
            - true_fc: true values of Fc. Fc(rho, grad_rho)
            - true_fc_gradsig: true gradients of Fc. dFc/drho, dFc/dsigma
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()
    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grho_flat = (2 * s_flat * k_F * rho_flat)
    sigma_flat = grho_flat**2

    # Create input array
    grho_inputs = jnp.stack([rho_flat, grho_flat], axis=1)

    # Calculate true Fc
    true_fc = Fc(rho_flat, grho_flat)
    if erho:
        def FcSIG_erho(rho, sigma):
            e_x_ueg = pw92c_unpolarized_scalar(rho)
            return FcSIG(rho, sigma) * rho * e_x_ueg
        true_fc_gradsig = jax.vmap(
            jax.grad(FcSIG_erho, argnums=(0, 1)))(rho_flat, sigma_flat)
    else:
        true_fc_gradsig = jax.vmap(
            jax.grad(FcSIG, argnums=(0, 1)))(rho_flat, sigma_flat)
    if return_sigma:
        sig_inputs = jnp.stack([rho_flat, sigma_flat], axis=1)
        return sig_inputs, true_fc, true_fc_gradsig
    else:
        return grho_inputs, true_fc, true_fc_gradsig


def gtv_FX_gridSIG(npts: int, s_lims: list[float], rho_lims: list[float],
                   separe_s: bool = False,
                   separe_s_val: float = 0.7, lim_s: float = 1.0,
                   slog: bool = False) \
                            -> tuple:
    """Get the true values, gradients and the inputs for Fx.

    The gradients are computed wrt rho and sigma.
    rho, sigma = inputs
    TODO: DOC
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points

    if slog:
        if separe_s:
            s_values_low = jnp.logspace(jnp.log10(START_S), jnp.log10(lim_s),
                                        num=int(separe_s_val * in_p))
            s_values_high = jnp.logspace(jnp.log10(lim_s), jnp.log10(STOP_S),
                                         num=int(in_p)-int(separe_s_val * in_p))
            s_values = jnp.concatenate([s_values_low, s_values_high])
        else:
            s_values = jnp.logspace(jnp.log10(START_S), jnp.log10(STOP_S),
                                     num=int(in_p))
    else:
        if separe_s:
            s_values_low = jnp.linspace(START_S, lim_s,
                                        num=int(separe_s_val * in_p))
            s_values_high = jnp.linspace(lim_s, STOP_S,
                                        num=int(in_p)-int(separe_s_val * in_p))
            s_values = jnp.concatenate([s_values_low, s_values_high])
        else:
            #  Unbiased sampling
            s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()
    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    sigma_flat = (2 * s_flat * k_F * rho_flat)**(2)

    # Create input array
    inputs = jnp.stack([rho_flat, sigma_flat], axis=1)

    # Calculate true Fx and Fc
    true_fx = FxSIG(rho_flat, sigma_flat)
    true_fx_gradsig = jax.vmap(
        jax.grad(FxSIG, argnums=(0, 1)))(rho_flat, sigma_flat)

    return inputs, true_fx, true_fx_gradsig


def gtv_FXC_grid_s_VO(npts: int, s_lims: list[float], rho_lims: list[float],
                      separe_s: bool = False,
                      separe_s_val: float = 0.7, lim_s: float = 1.0,
                      Fx_ref=None, Fc_ref=None) \
                             -> tuple:
    """Get the true values and the inputs for Fx and Fc.

    Being Fx the exchange enchancement factor. It retuns the values of Fx and
    Fc, and the gradients (w.r.t. the density and the derivative of the
    density), for both cases, in an uniform grid, that is defined by the
    inputs of the functions. Note that these inputs are also returned.
    VO = Values Only.
    TODO update
    Args:
        npts (int): [Description of npts].
        s_lims (list[float]): [Description of s_lims].
        rho_lims (list[float]): [Description of rho_lims].
        separe_s (bool, optional): [Description of separe_s]. Defaults to False.
        separe_s_val (float, optional): [Description of separe_s_val]. Defaults to 0.7.
        lim_s (float, optional): [Description of lim_s]. Defaults to 1.0.

    Returns:
        tuple:
            - sinputs: [Description of inputs].
            - true_fx_s: [Description of true_fc].
            - true_fc_s: [Description of true_fc].
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s
    if Fx_ref is None:
        Fx_ref = Fx
    if Fc_ref is None:
        Fc_ref = Fc

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()
    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    # Create input array
    sinputs = jnp.stack([rho_flat, s_flat], axis=1)

    # Calculate true Fx and Fc
    true_fx_s = Fx_ref(rho_flat, grad_rho_flat)
    true_fc_s = Fc_ref(rho_flat, grad_rho_flat)

    return sinputs, true_fx_s, true_fc_s


def get_true_values_FC_grid_s(npts: int,
                              s_lims: list[float], rho_lims: list[float],
                              separe_s: bool = False,
                              separe_s_val: float = 0.7, lim_s: float = 1.0) \
                             -> tuple:
    """Get the true values and the inputs for Fc.

    Being Fc the correlation enchancement factor. It retuns the values of Fc
    and the gradients (w.r.t. the density and the derivative of the density)
    in an uniform grid, that is defined by the inputs of the functions.
    Note that these inputs are also returned.
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Create input array
    inputs = jnp.stack([rho_flat, s_flat], axis=1)

    # Calculate true Fx and Fc values
    true_fc = Fc_s(rho_flat, s_flat)

    def scalar_Fc(rho, grad_rho):
        return Fc(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
    true_fc_grad = grad_Fc(rho_flat, s_flat)

    return inputs, true_fc, true_fc_grad


def gtv_FC_gridSIG(npts: int, s_lims: list[float], rho_lims: list[float],
                   separe_s: bool = False,
                   separe_s_val: float = 0.7, lim_s: float = 1.0,
                   slog= False) \
                            -> tuple:
    """Get the true values, gradients and the inputs for Fc.

    The gradients are computed wrt rho and sigma.
    rho, sigma = inputs
    TODO: DOC
    """
    # Suggest values for the hyperparameters
    # Note: Number of grid points [i.e. (x,y) pairs] is actually sqrt(NPTS).
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if slog:
        if separe_s:
            s_values_low = jnp.logspace(jnp.log10(START_S), jnp.log10(lim_s),
                                         num=int(separe_s_val * in_p))
            s_values_high = jnp.logspace(jnp.log10(lim_s), jnp.log10(STOP_S),
                                          num=int(in_p)-int(separe_s_val * in_p))
        else:
            s_values = jnp.logspace(jnp.log10(START_S), jnp.log10(STOP_S),
                                     num=int(in_p))
    else:
        if separe_s:
            s_values_low = jnp.linspace(START_S, lim_s,
                                        num=int(separe_s_val * in_p))
            s_values_high = jnp.linspace(lim_s, STOP_S,
                                        num=int(in_p)-int(separe_s_val * in_p))
            s_values = jnp.concatenate([s_values_low, s_values_high])
        else:
            #  Unbiased sampling
            s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()
    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    sigma_flat = (2 * s_flat * k_F * rho_flat)**(2)

    # Create input array
    inputs = jnp.stack([rho_flat, sigma_flat], axis=1)

    # Calculate true Fx and Fc
    true_fc = FcSIG(rho_flat, sigma_flat)
    true_fc_gradsig = jax.vmap(
        jax.grad(FcSIG, argnums=(0, 1)))(rho_flat, sigma_flat)

    return inputs, true_fc, true_fc_gradsig


def get_uniform_grids(npts: int,
                      s_lims: list[float], rho_lims: list[float]) \
                      -> tuple:
    NPTS = npts
    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))
    s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    return rho_flat, grad_rho_flat


def gtv_from_rho_and_s(rho_values, s_values, F_function: callable, return_val='grad_rho') \
                            -> tuple:
    """Get the true values and the inputs for training the network.

    TODO: derivatives

    Args:
        rho_values (jnp.ndarray): Array of density values.
        s_values (jnp.ndarray): Array of reduced gradient values.
        F_function (callable): Function to compute Fx/Fc given rho and grad_rho.
        return_val (str, optional): Whether to return 'grad_rho', 'sigma', or 's'
            (check the inputs your function expects). Defaults to 'grad_rho'.
    Returns:
        tuple:
            - inputs (jnp.ndarray): Array of shape (N, 2) with inputs for the model.
                rho and grad_rho/sigma/s (depending on return_val).
            - true_f (jnp.ndarray): Array of shape (N,) with true Fx/Fc values
                (depending on F_function)
    """
    rho_flat = jnp.array(rho_values).flatten()
    s_flat = jnp.array(s_values).flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    # Calculate true Fx
    true_f = F_function(rho_flat, grad_rho_flat)
    if return_val == 'grho':
        return_val = 'grad_rho'
    # Create input array
    if return_val == 'grad_rho':
        inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)
    elif return_val == 'sigma':
        sigma_flat = grad_rho_flat**2
        inputs = jnp.stack([rho_flat, sigma_flat], axis=1)
    elif return_val == 's':
        inputs = jnp.stack([rho_flat, s_flat], axis=1)
    else:
        raise ValueError(f"return_val must be 'grad_rho', 'sigma', or 's', got {return_val}")
    return inputs, true_f


def get_true_values_FC_water(grid_level: int = 3, filter_s: bool = False,
                             smax: float = 100.0,
                             extrapoints: Union[None, jt.ArrayLike] = None,
                             plot_values: bool = True):
    """Get true values for FC in the water grid.

    It takes the rho and grad_rho values from pyscf (using the grid already
    defined by pyscf) and returns those input values, and the values of
    Fc (correlation enhancement factor) and grad_Fc in those grid points.
    Args:
        grid_level (int): defined the level of the pyscf grid (mf.grids.level).
        filter_s (bool): if True, it sets the maximum value of s to be smax.
            It thus deletes all the coords/rhos corresponding to s>smax.
        smax (float): maximum value of s if filter_s=True.
        extrapoints (jt.ArrayLike[float], (N, 2)): array with extra points
            to add to the inputs. Extrapoints[:, 0] corresponds to the rhos,
            and extrapoints[:, 1] to grad_rho. Default: None (No points added)
        plot_values (bool): whether to plot the histograms of the selected
            rhos and s as inputs. Default: true.
    Returns:
        tuple:
            - inputs (jt.Arraylike[float], (Npoints, 2)): inputs used
                evaluating Fc. Depends on the args given.
            - true_fc ((jt.Arraylike[float], (Npoints))): Fc evaluated in
                inputs. True values obtained from Fc function.
    """
    mol = gto.M(
        atom='''
        O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587 ''',
        basis='ccpvdz')
    mf = dft.RKS(mol)

    # Define grid level that defines the coords to evaluate
    mf.grids.level = grid_level
    mf.kernel()
    # Get density matrix and the number integration object
    dm = mf.make_rdm1()
    coords = mf.grids.coords
    ao = dft.numint.eval_ao(mol, coords, deriv=1)

    # Get the density and gradients in the coords
    rho_out = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
    print(f'Evaluating the grid in {rho_out.shape[1]} points')

    # Get the values and the gradients
    rho = jnp.array(rho_out[0, :])
    grad_rho = jnp.array(LA.norm(rho_out[1:, :], axis=0))

    if extrapoints is not None:
        rho = jnp.concatenate([rho, extrapoints[:, 0]])
        grad_rho = jnp.concatenate([grad_rho, extrapoints[:, 1]])
    s = S(rho, grad_rho)

    # If asked, quit the coords that are higher than dertain point
    if filter_s:
        mask = jnp.where(s < smax)
        s = s[mask]
        rho = rho[mask]
        grad_rho = grad_rho[mask]

    if plot_values:
        plot_rho_s_hist(rho, s, smax)
    # Create input array
    inputs = jnp.stack([rho, grad_rho], axis=1)

    # Calculate true Fx and Fc values
    true_fc = Fc(rho, grad_rho)

    return inputs, true_fc


def get_true_values_FX_grid_general(npts: int, input_var,
                                    s_lims: list[float], rho_lims: list[float],
                                    deriv_wrt=None,
                                    separe_s: bool = False,
                                    separe_s_val: float = 0.7,
                                    lim_s: float = 1.0,
                                    erho: bool = False) \
                                    -> tuple:
    """Get the true values and the inputs for Fx.

    input_var: The variable we want to use as input of our network.
    deriv_wrt: With respect to which variable we want the derivative.

    TODO: DOC, TEST
    """
    variables = ['grad_rho', 's', 'sigma', None]
    if input_var not in variables:
        raise ValueError(f"""input_var must be one of {variables},
but got {input_var}""")
    if deriv_wrt not in variables:
        raise ValueError(f"""deriv_wrt must be one of {variables},
but got {deriv_wrt}""")
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    if input_var == 's':
        inputs = jnp.stack([rho_flat, s_flat], axis=1)
    elif input_var == 'sigma':
        sigma_flat = grad_rho_flat**2
        inputs = jnp.stack([rho_flat, sigma_flat], axis=1)
    elif input_var == 'grad_rho':
        inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)
    # Calculate true Fx values
    true_fx = Fx(rho_flat, grad_rho_flat)
    if deriv_wrt is None:
        print(f'''Returning inputs = (rho, {input_var}), true_fx
WARNING: No derivative requested''')
        return inputs, true_fx, None
    else:
        this_fx = {
            'grad_rho': Fx,
            's': Fx_s_2inputs,
            'sigma': FxSIG
        }.get(deriv_wrt)
        if deriv_wrt == 's':
            inp_deriv = s_flat
        elif deriv_wrt == 'sigma':
            sigma_flat = grad_rho_flat**2
        elif deriv_wrt == 'grad_rho':
            inp_deriv = grad_rho_flat
        if erho:
            def Fx_erho(rho, inp_deriv):
                e_x_ueg = lda_x(rho)
                return this_fx(rho, inp_deriv) * rho * e_x_ueg
            true_fx_grad = jax.vmap(
                jax.grad(Fx_erho, argnums=(0, 1)))(rho_flat, inp_deriv)
        else:
            true_fx_grad = jax.vmap(
                jax.grad(this_fx, argnums=(0, 1)))(rho_flat, inp_deriv)
        print(f'''Returning inputs = (rho, {input_var}), true_fx and
true_fx_grad (derivative w.r.t. [rho, {deriv_wrt}])''')
        return inputs, true_fx, true_fx_grad


def get_true_values_FC_grid_general(npts: int, input_var,
                                    s_lims: list[float], rho_lims: list[float],
                                    deriv_wrt=None,
                                    separe_s: bool = False,
                                    separe_s_val: float = 0.7, lim_s: float = 1.0,
                                    erho: bool =False) \
                                    -> tuple:
    """Get the true values and the inputs for Fc.

    input_var: The variable we want to use as input of our network.
    deriv_wrt: With respect to which variable we want the derivative.

    TODO: DOC, TEST
    """
    variables = ['grad_rho', 's', 'sigma', None]
    if input_var is None:
        raise ValueError(f"input_var must be one of {variables[:3]}, but got {input_var}")
    if input_var not in variables:
        raise ValueError(f"input_var must be one of {variables[:3]}, but got {input_var}")
    if deriv_wrt not in variables:
        raise ValueError(f"deriv_wrt must be one of {variables}, but got {deriv_wrt}")
    NPTS = npts

    START_S, STOP_S = s_lims  # Adjust these values as needed
    START_RHO, STOP_RHO = rho_lims  # Range of rho values to test
    # Generate training data
    # Create more points for small values of s

    in_p = jnp.sqrt(NPTS)  # Input points
    if separe_s:
        s_values_low = jnp.linspace(START_S, lim_s,
                                    num=int(separe_s_val * in_p))
        s_values_high = jnp.linspace(lim_s, STOP_S,
                                     num=int(in_p)-int(separe_s_val * in_p))
        s_values = jnp.concatenate([s_values_low, s_values_high])
    else:
        #  Unbiased sampling
        s_values = jnp.linspace(START_S, STOP_S, num=int(in_p))

    rho_values = jnp.logspace(jnp.log10(START_RHO),
                              jnp.log10(STOP_RHO), num=int(in_p))

    rho_mesh, s_mesh = jnp.meshgrid(rho_values, s_values)
    rho_flat = rho_mesh.flatten()
    s_flat = s_mesh.flatten()

    # Calculate grad_rho
    k_F = (3 * jnp.pi**2 * rho_flat)**(1/3)
    grad_rho_flat = 2 * s_flat * k_F * rho_flat

    if input_var == 's':
        inputs = jnp.stack([rho_flat, s_flat], axis=1)
    elif input_var == 'sigma':
        sigma_flat = grad_rho_flat**2
        inputs = jnp.stack([rho_flat, sigma_flat], axis=1)
    elif input_var == 'grad_rho':
        inputs = jnp.stack([rho_flat, grad_rho_flat], axis=1)

    true_fc = Fc(rho_flat, grad_rho_flat)

    if deriv_wrt is None:
        print(f'''Returning inputs = (rho, {input_var}), true_fc
WARNING: No derivative requested''')
        return inputs, true_fc, None
    else:
        this_fc = {
            'grad_rho': Fc,
            's': Fc_s,
            'sigma': FcSIG
        }.get(deriv_wrt)

        if deriv_wrt == 's':
            inp_deriv = s_flat
        elif deriv_wrt == 'sigma':
            sigma_flat = jnp.sqrt(grad_rho_flat)
        elif deriv_wrt == 'grad_rho':
            inp_deriv = grad_rho_flat

        def scalar_Fc(rho, inp_deriv):
            return this_fc(jnp.array([rho]), jnp.array([inp_deriv]))[0]

        if erho:
            def Fc_erho(rho, inp_deriv):
                e_x_ueg = pw92c_unpolarized_scalar(rho)
                return scalar_Fc(rho, inp_deriv) * rho * e_x_ueg
            true_fc_grad = jax.vmap(
                jax.grad(Fc_erho, argnums=(0, 1)))(rho_flat, inp_deriv)
        else:
            true_fc_grad = jax.vmap(
                jax.grad(scalar_Fc, argnums=(0, 1)))(rho_flat, inp_deriv)
        if not erho:
            print(f'''Returning inputs = (rho, {input_var}), true_fc and
true_fc_grad (derivative w.r.t. [rho, {deriv_wrt}])''')
        else:
            print(f'''Returning inputs = (rho, {input_var}), true_fc and
true_fc_grad (derivative of rho*Fc w.r.t. [rho, {deriv_wrt}])''')
        return inputs, true_fc, true_fc_grad

# --------- TRUE ENHANCEMENT FACTORS --------------------


def Fx(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fx
    """
    s = S(rho, grad_rho)
    kappa, mu = _PBE_KAPPA, _PBE_MU
    # Exchange enhancement factor
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)

    return Fx


def Fx_packed(inputs) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        inputs (tuple): (rho, grad_rho)

    Returns:
        jt.ArrayLike: Fx
    """
    rho, grad_rho = inputs

    return Fx(rho, grad_rho)


def Fx_packedSIG(inputs) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        inputs (tuple): (rho, grad_rho)

    Returns:
        jt.ArrayLike: Fx
    """
    rho, sigma = inputs

    return FxSIG(rho, sigma)

def Fx_packedSIG_minrho(inputs) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        inputs (tuple): (rho, grad_rho)

    Returns:
        jt.ArrayLike: Fx
    """
    rho, sigma = inputs
    rho = jnp.maximum(1e-14, rho)  # Prevents division by 0

    return FxSIG(rho, sigma)


def Fx_minrho(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and ∇n.
    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fx
    """
    rho = jnp.maximum(1e-12, rho)  # Prevents division by 0
    s = S(rho, grad_rho)
    kappa, mu = _PBE_KAPPA, _PBE_MU
    # Exchange enhancement factor
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)

    return Fx


def Fx_s(s: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor as a function of s.

    Args:
        s (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fx
    """
    kappa, mu = _PBE_KAPPA, _PBE_MU

    # Exchange enhancement factor
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
    return Fx


def Fx_s_2inputs(_,  s: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor as a function of s.

    Args:
        s (jt.ArrayLike): s values.

    Returns:
        jt.ArrayLike: Fx
    """
    kappa, mu = _PBE_KAPPA, _PBE_MU

    # Exchange enhancement factor
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
    return Fx


def FxSIG(rho: jt.ArrayLike, sigma: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference exchange enhancement factor.

    As a function of n and sigma.
    Args:
        rho (jt.ArrayLike): 
        sigma (jt.ArrayLike):

    Returns:
        jt.ArrayLike: Fx
    """
    kf = (3 * jnp.pi**2 * rho)**(1/3)
    s = jnp.sqrt(sigma)/(2*kf*rho)
    kappa, mu = _PBE_KAPPA, _PBE_MU
    # Exchange enhancement factor
    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)

    return Fx


def Fc(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is taken for PW LDA (see pw92c_unpolarized),
    assuming zeta=0.

    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def Fc_packed(inputs) -> jt.ArrayLike:
    """Exact reference correlation enhancement factor for PBE.

    It expects as inputs (rho, grad_rho).

    Args:
        inputs (tuple): (rho, grad_rho)
    Returns:
        jt.ArrayLike: Fc
    """

    rho, grad_rho = inputs

    return Fc_scalar(rho, grad_rho)


def Fc_packedSIG_minrho(inputs) -> jt.ArrayLike:
    rho, sigma = inputs
    rho = jnp.maximum(1e-24, rho)  # Prevents division by 0
    grad_rho = jnp.sqrt(sigma)
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized_scalar(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc
def Fc_packedSIG_minrho_14(inputs) -> jt.ArrayLike:
    rho, sigma = inputs
    rho = jnp.maximum(1e-14, rho)  # Prevents division by 0
    grad_rho = jnp.sqrt(sigma)
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized_scalar(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc

def Fc_packedSIG(inputs) -> jt.ArrayLike:

    rho, sigma = inputs
    grad_rho = jnp.sqrt(sigma)

    return Fc_scalar(rho, grad_rho)


def Fc_scalar(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is taken for PW LDA (see pw92c_unpolarized),
    assuming zeta=0.

    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized_scalar(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def Fc_minrho(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is taken for PW LDA (see pw92c_unpolarized),
    assuming zeta=0.

    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    rho = jnp.maximum(1e-12, rho)  # Prevents division by 0
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def Fc_s(rho: jt.ArrayLike, s: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is taken for PW LDA (see pw92c_unpolarized),
    assuming zeta=0.
    As a function of rho and s.

    Args:
        rho (jt.ArrayLike): 
        s (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    rho = jnp.maximum(1e-12, rho)  # Prevents division by 0
    grad_rho = Grad_rho(rho, s)

    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def FcSIG(rho: jt.ArrayLike, sigma: jt.ArrayLike) -> jt.ArrayLike:
    """Exact reference correlation enhancement function, as a function of sigma
    TODO DOC"""

    grad_rho = jnp.sqrt(sigma)
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Use pw92c_unpolarized for e_heg_c
    e_heg_c = pw92c_unpolarized_scalar(rho)

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def Fc_pyscf(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) \
        -> jt.ArrayLike:
    """Exact reference correlation enhancement function.

    The energy per unit electron is calculated with pyscf, assuming zeta=0.

    Args:
        rho (jt.ArrayLike): 
        s (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: Fc
    """
    rho = jnp.maximum(1e-12, rho)  # Prevents division by 0
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = _PBE_BETA
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Calculate e_heg_c (heterogeneous electron gas correlation energy)
    e_heg_c = dft.libxc.eval_xc('LDA_C_PW,', rho, spin=0, deriv=1)[0]

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 *
                        ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # Correlation enhancement factor

    return Fc


# ---------- FUNCTIONALS -------------------


def pw92c_unpolarized(rho: jt.ArrayLike) -> jt.ArrayLike:
    """
    Implements the Perdew-Wang '92 local correlation (beyond RPA)
    for the unpolarized case.
    Reference: J.P.Perdew & Y.Wang, PRB, 45, 13244 (1992)

    Parameters:
    rho : jax.numpy array
        Total electron density array on a grid.

    Returns:
    EC : jax.numpy array
        Correlation energy density array.
    """
    # Ensure rho is a jax.numpy array
    rho = jnp.asarray(rho)

    # Parameters from Table I of Perdew & Wang, PRB, 45, 13244 (92)
    A = jnp.array([0.031090690869654895, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370, 0.20548, 0.11125])
    BETA1 = jnp.array([7.5957, 14.1189, 10.357])
    BETA2 = jnp.array([3.5876, 6.1977, 3.6231])
    BETA3 = jnp.array([1.6382, 3.3662, 0.88026])
    BETA4 = jnp.array([0.49294, 0.62517, 0.49671])

    # Compute rs (Wigner-Seitz radius) for each grid point
    rs = (3 / (4 * jnp.pi * rho))**(1/3)

    # Compute G for unpolarized case (zeta = 0) across all grid points
    def compute_g(rs):
        G = jnp.zeros((len(rs), 3))
        for k in range(3):
            B = (BETA1[k] * jnp.sqrt(rs) +
                 BETA2[k] * rs +
                 BETA3[k] * rs**1.5 +
                 BETA4[k] * rs**2)
            C = 1 + 1 / (2 * A[k] * B)
            G = G.at[:, k].set(-2 * A[k] * (1 + ALPHA1[k] * rs) * jnp.log(C))
        return G

    # Apply compute_g to each grid point
    G = compute_g(rs)

    # For unpolarized case, correlation energy density is G[0]
    EC = G[:, 0]

    return EC


def pw92c_unpolarized_scalar(rho):
    # Ensure rho is a jax.numpy array
    rho = jnp.asarray(rho)

    # Parameters from Table I of Perdew & Wang, PRB, 45, 13244 (92)
    A = jnp.array([0.031090690869654895, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370, 0.20548, 0.11125])
    BETA1 = jnp.array([7.5957, 14.1189, 10.357])
    BETA2 = jnp.array([3.5876, 6.1977, 3.6231])
    BETA3 = jnp.array([1.6382, 3.3662, 0.88026])
    BETA4 = jnp.array([0.49294, 0.62517, 0.49671])

    # Compute rs (Wigner-Seitz radius) for each grid point
    rs = (3 / (4 * jnp.pi * rho))**(1/3)
    # Convert rs to an array if it is a scalar
    rs = jnp.atleast_1d(rs)

    # Compute G for unpolarized case (zeta = 0) across all grid points
    def compute_g(rs):
        G = jnp.zeros((len(rs), 3))
        for k in range(3):
            B = (BETA1[k] * jnp.sqrt(rs) +
                 BETA2[k] * rs +
                 BETA3[k] * rs**1.5 +
                 BETA4[k] * rs**2)
            C = 1 + 1 / (2 * A[k] * B)
            G = G.at[:, k].set(-2 * A[k] * (1 + ALPHA1[k] * rs) * jnp.log(C))
        return G if rs.shape[0] > 1 else G[0]

    # Apply compute_g to each grid point
    G = compute_g(rs)

    # For unpolarized case, correlation energy density is G[0]
    EC = G[:, 0] if rs.shape[0] > 1 else G[0]

    return EC


def lda_x(rho):
    return -3/4*(3/jnp.pi)**(1/3)*jnp.sign(rho)*(jnp.abs(rho))**(1 / 3)


def jsexchng_exonly(ds, irel=0, nsp=1):
    CO14 = 0.014
    TFTM = 2**(4/3)-2
    A0 = (4/(9*jnp.pi))**(1/3)
    ALP = 2/3

    if nsp == 2:
        pass
        #  worry about polarization later
    else:
        try:
            D = ds[0]
        except:
            D = ds
        
        Z = 0
        FZ = 0
        FZP = 0
    RS = (3/ (4*jnp.pi*D) )**(1/3)
    VXP = -(3*ALP/(2*jnp.pi*A0*RS))
    EXP_VAR = 3*VXP/4
    if (irel == 1):
        pass
    VXF = 2**(1/3)*VXP
    EXF = 2**(1/3)*EXP_VAR
    if nsp == 2:
        pass
    else:
        VX = VXP
        EX = EXP_VAR
    return EX

# --------------- CONVERSIONS --------------------
def S_general(rho: jt.ArrayLike, inp2: jt.ArrayLike, inp_var: str) -> jt.ArrayLike:
    """
    Args:
        rho (jt.ArrayLike): 
        inp2 (jt.ArrayLike): grad_rho or sigma
        inp_var (str): 'grad_rho' or 'sigma'

    Returns:
        jt.ArrayLike: s
    """
    if inp_var == 'grad_rho':
        grad_rho = inp2
    elif inp_var == 'sigma':
        grad_rho = jnp.sqrt(inp2)
    else:
        raise ValueError(f"inp_var must be 'grad_rho' or 'sigma', but got {inp_var}")

    k_F = (3 * jnp.pi**2 * rho)**(1/3)
    s = grad_rho / (2 * k_F * rho)
    return s


def S(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """
    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: s
    """
    # rho = jnp.maximum(1e-12, rho)  # Prevents division by 0
    k_F = (3 * jnp.pi**2 * rho)**(1/3)
    s = grad_rho / (2 * k_F * rho)
    return s

def S_prevent0(rho: jt.ArrayLike, grad_rho: jt.ArrayLike) -> jt.ArrayLike:
    """Is s but where rho=0, s=0 to prevent division by zero.
    Args:
        rho (jt.ArrayLike): 
        grad_rho (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: s
    """
    k_F = (3 * jnp.pi**2 * rho)**(1/3)
    safe_rho = jnp.where(rho > 1e-14, rho, 1.0)  # avoid dividing by zero
    safe_kF = (3 * jnp.pi**2 * safe_rho)**(1/3)

    # Avoid division by 0 putting s=0 where rho=0
    s = jnp.where(rho > 1e-14, grad_rho / (2 * safe_kF * safe_rho), 0.0)
    return s


def S_np(rho: npt.NDArray, grad_rho: npt.NDArray) -> npt.ArrayLike:
    """
    Args:
        rho (npt.NDArray): 
        grad_rho (npt.NDArray): 

    Returns:
        npt.ArrayLike: s
    """
    rho = np.maximum(1e-12, rho)  # Prevents division by 0
    k_F = (3 * np.pi**2 * rho)**(1/3)
    s = grad_rho / (2 * k_F * rho)
    return s


def Grad_rho(rho: jt.ArrayLike, s: jt.ArrayLike) -> jt.ArrayLike:
    """
    Args:
        rho (jt.ArrayLike): 
        s (jt.ArrayLike): 

    Returns:
        jt.ArrayLike: grad_rho 
    """
    
    k_F = (3 * np.pi**2 * rho)**(1/3)
    grad_rho = 2 * s * k_F * rho
    return grad_rho

# Utils


def plot_rho_s_hist(rho: jt.ArrayLike, s: jt.ArrayLike, smax: float):
    """
    Args:
        rho (jt.ArrayLike): 
        s (jt.ArrayLike): 
        smax (float): maximum value of s.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 4))
    axes[0, 0].hist(s, np.arange(0, s.max(), 1), color='purple')
    axes[0, 0].set_ylim(0, 200)
    axes[0, 1].hist(s, np.arange(0, 5, 0.05), color='purple')
    axes[1, 0].hist(s, np.arange(0, 10, 0.1), color='purple')
    axes[1, 1].hist(s, np.arange(10, smax*1.1, 1), color='purple')
    axes[1, 2].hist(rho, np.arange(0, rho.max(), 1), color='g')
    axes[1, 2].set_ylim(0, 60)
    axes[0, 2].hist(rho, np.arange(0, 5, 0.05), color='g')
    plt.show()
