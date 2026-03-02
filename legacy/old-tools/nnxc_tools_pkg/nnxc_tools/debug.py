import numpy as np
import jax

import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # Enable 64 bit precision


from functools import partial
import warnings
import os
from modules.custom_xc import custom_potential_ft_ec_sc, custom_potential_ft_ex_sc,\
    custom_potential_ft_ec_sc_SIG, custom_potential_ft_ex_sc_SIG, \
    custom_potential_ft_ec_sc_s, custom_potential_ft_ex_sc_s


######## COMMON FUNCTIONS #########

def get_rho_sigma_polarized(rho):
    rho_u, rho_d = rho
    rho0u, dxu, dyu, dzu = rho_u[:4]
    rho0d, dxd, dyd, dzd = rho_d[:4]
    dxu2 = dxu*dxu
    dyu2 = dyu*dyu
    dzu2 = dzu*dzu
    dxud = dxu*dxd
    dyud = dyu*dyd
    dzud = dzu*dzd
    dxd2 = dxd*dxd
    dyd2 = dyd*dyd
    dzd2 = dzd*dzd
    sigma1 = dxu2+dyu2+dzu2  # sigma up up
    sigma2 = dxud+dyud+dzud # sigma up down
    sigma3 = dxd2+dyd2+dzd2 # sigma down down
    return rho0u, rho0d, sigma1, sigma2, sigma3


######## DEBUG FUNCTIONS #########

def get_my_eval_xc_ft_DEBUG(model_fx, model_fc, sigma=False, polarized=True):
    """
    If sigma is True, the model expects sigma as input.
    If polarized is True, the functional can also be used in spin-polarized calculations.
    """
    if sigma:
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc_SIG, model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc_SIG, model_fc=model_fc)
    else:
        try:
            model_name_lower_fx = model_fx.name.lower()
            model_name_lower_fc = model_fc.name.lower()
            if 'sig' in model_name_lower_fx or 'sig' in model_name_lower_fc:
                warnings.warn("""CHECK YOUR NETWORKS! The model is trained
with sigma as input. Put sigma=True""")
        except AttributeError:
            print('Your functional has not atribute name. If it is a function,\
you may continue safely. If it is a trained functional, be carefull as a lot\
of functions expect a .name attribute.')
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc, model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc, model_fc=model_fc)
    if polarized:
        my_eval_xc_ft = partial(eval_xc_GGA_pol_general_DEBUG, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    else:
        my_eval_xc_ft = partial(eval_xc_PBE_general_DEBUG, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    return my_eval_xc_ft

def eval_xc_GGA_pol_general_DEBUG(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                            custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.
    It takes into account polarization, but we do not calculate the hessian"""
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    try:
        rhoshape = len(rho.shape)
        pol = 3
    except:
        rhoshape = len(rho)
        pol = 2
    if rhoshape != pol:
        exc, vxc, fxc, kxc = eval_xc_PBE_general_DEBUG(xc_code, rho, spin, relativity, deriv, omega, verbose, custom_ex=custom_ex, custom_ec=custom_ec)
    else:
        exc, vxc, fxc, kxc = eval_xc_GGA_pol_sara_DEBUG(xc_code, rho, spin, relativity, deriv, omega, verbose,
                                                  custom_ex=custom_ex, custom_ec=custom_ec)
    return exc, vxc, fxc, kxc

def eval_xc_PBE_general_DEBUG(xc_code, rho, spin=0, relativity=0, deriv=1,
                        omega=None, verbose=None,
                        custom_ex=None, custom_ec=None):
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    try:
        rho0, dx, dy, dz = rho[:4]
        sigma = jnp.array(dx**2+dy**2+dz**2)
    except ValueError:
        rho0, drho = rho[:4]
        sigma = jnp.array(drho**2)
    rho0 = jnp.asarray(rho0)
    fx = custom_ex(rho0, sigma)
    fc = custom_ec(rho0, sigma)
    pid = os.getpid()
    if jnp.isnan(fx).any().item() or jnp.isinf(fx).any().item():
        with open(f"debug_nan_inf_{pid}.txt", "a") as f:
            f.write(f"NaN/Inf in fx, shape={fx.shape}\n")
            f.write(f"rho0 min/max: {jnp.min(rho0)}/{jnp.max(rho0)}\n")
            f.write(f"sigma min/max: {jnp.min(sigma)}/{jnp.max(sigma)}\n")
    if jnp.isnan(fc).any().item() or jnp.isinf(fc).any().item():
        with open(f"debug_nan_inf_{pid}.txt", "a") as f:
            f.write(f"NaN/Inf in fc, shape={fc.shape}\n")
            f.write(f"rho0 min/max: {jnp.min(rho0)}/{jnp.max(rho0)}\n")
            f.write(f"sigma min/max: {jnp.min(sigma)}/{jnp.max(sigma)}\n")
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)
    def rho_times_func(rho0, sigma):
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrho, vsigma = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1)))(rho0, sigma)
    exc = np.asarray(exc)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)
    vxc = (vrho, vsigma, None, None)
    fxc = None
    kxc = None
    return exc, vxc, fxc, kxc

def eval_xc_GGA_pol_sara_DEBUG(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                         custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.
    It takes into account polarization, but we do not calculate the hessian
    """
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    rho0u, rho0d, sigma1, sigma2, sigma3 = get_rho_sigma_polarized(rho)
    rho0 = jnp.array(rho0u+rho0d)
    sigma = sigma1+sigma2+sigma3
    sigma = jnp.array(sigma)
    fx = custom_ex(rho0, sigma)
    fc = custom_ec(rho0, sigma)
    pid = os.getpid()
    if jnp.isnan(fx).any().item() or jnp.isinf(fx).any().item():
        with open(f"debug_nan_inf_{pid}.txt", "a") as f:
            f.write(f"NaN/Inf in fx, shape={fx.shape}\n")
            f.write(f"rho0 min/max: {jnp.min(rho0)}/{jnp.max(rho0)}\n")
            f.write(f"sigma min/max: {jnp.min(sigma)}/{jnp.max(sigma)}\n")
    if jnp.isnan(fc).any().item() or jnp.isinf(fc).any().item():
        with open(f"debug_nan_inf_{pid}.txt", "a") as f:
            f.write(f"NaN/Inf in fc, shape={fc.shape}\n")
            f.write(f"rho0 min/max: {jnp.min(rho0)}/{jnp.max(rho0)}\n")
            f.write(f"sigma min/max: {jnp.min(sigma)}/{jnp.max(sigma)}\n")
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)
    def rho_times_func(rho0u, rho0d, sigma1, sigma2, sigma3):
        rho0 = jnp.array(rho0u+rho0d)
        sigma = sigma1+sigma2+sigma3
        sigma = jnp.array(sigma)
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrhou, vrhod, vsigma1, vsigma2, vsigma3 = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1, 2, 3, 4)))(rho0u, rho0d, sigma1, sigma2, sigma3)
    vrho = jnp.vstack((vrhou, vrhod))
    vsigma = jnp.vstack((vsigma1, vsigma2, vsigma3))
    vxc = (vrho, vsigma)
    fxc = None
    kxc = None
    TRANSPOSE = True
    if TRANSPOSE:
        vxc = [i.T for i in vxc]
    return exc, vxc, fxc, kxc


################ STABILITY FUNCTIONS ################

def get_my_eval_xc_ft_STABLE(model_fx, model_fc, sigma=False,
                             polarized=True, input_param='grho'):
    """
    If sigma is True, the model expects sigma as input.
    If polarized is True, the functional can also be used in spin-polarized calculations.

    Input_param can be 's', 'grho' or 'sigma'.
    INPUT PARAM IS THE PARAMETER THAT MY NETWORK EXPECTS, but note that
    custom_potential_ft always embeds our NN, so expects SIGMA as input.
    """
    #
    if sigma and input_param != 'sigma':
        warnings.warn("You set sigma=True but input_param is not 'sigma'. Setting input_param to 'sigma'.")
        input_param = 'sigma'
    jax.config.update("jax_enable_x64", True)  # Enable 64 bit precision   

    try:
        model_name_lower_fx = model_fx.name.lower()
        model_name_lower_fc = model_fc.name.lower()
        if 'sig' in model_name_lower_fx or 'sig' in model_name_lower_fc:
            if input_param != 'sigma' and not sigma:
                warnings.warn("""CHECK YOUR NETWORKS! The model is trained
    with sigma as input. Put sigma=True""")
        elif 'fxg' in model_name_lower_fx or 'fcg' in model_name_lower_fc:
            if input_param != 'grho' and sigma:
                warnings.warn("""CHECK YOUR NETWORKS! The model is trained
    with grad rho as input. Put sigma=False or input_param='grho'""")
    except AttributeError:
        print('Your functional has not atribute name. If it is a function,\
you may continue safely. If it is a trained functional, be carefull as a lot\
of functions expect a .name attribute.')
        # Input param is grho by default
    if input_param == 'grho':
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc, model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc, model_fc=model_fc)
        print("[info] Using gradient of rho as input parameter")
    elif input_param == 'sigma':
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc_SIG, model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc_SIG, model_fc=model_fc)
        print("[info] Using sigma as input parameter")
    elif input_param == 's':
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc_s, model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc_s, model_fc=model_fc)
    if polarized:
        my_eval_xc_ft = partial(eval_xc_GGA_pol_general_STABLE, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    else:
        my_eval_xc_ft = partial(eval_xc_PBE_general_STABLE, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    return my_eval_xc_ft

def eval_xc_GGA_pol_general_STABLE(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                            custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.
    It takes into account polarization, but we do not calculate the hessian"""
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    try:
        rhoshape = len(rho.shape)
        pol = 3
    except:
        rhoshape = len(rho)
        pol = 2
    if rhoshape != pol:
        exc, vxc, fxc, kxc = eval_xc_PBE_general_STABLE(xc_code, rho, spin, relativity, deriv, omega, verbose, custom_ex=custom_ex, custom_ec=custom_ec)
    else:
        exc, vxc, fxc, kxc = eval_xc_GGA_pol_sara_STABLE(xc_code, rho, spin, relativity, deriv, omega, verbose,
                                                  custom_ex=custom_ex, custom_ec=custom_ec)
    return exc, vxc, fxc, kxc
def eval_xc_PBE_general_STABLE(xc_code, rho, spin=0, relativity=0, deriv=1,
                        omega=None, verbose=None,
                        custom_ex=None, custom_ec=None):
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    try:
        rho0, dx, dy, dz = rho[:4]
        sigma = jnp.array(dx**2+dy**2+dz**2)
    except ValueError:
        rho0, drho = rho[:4]
        sigma = jnp.array(drho**2)
    rho0 = jnp.asarray(rho0)
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)
    def rho_times_func(rho0, sigma):
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrho, vsigma = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1)))(rho0, sigma)
    
    # IN the places where rho0 is smaller than 1e-12, set vrho and vsigma to 0
    eps = 1e-14
    exc = jnp.where(jnp.abs(rho0) < eps, 0.0, exc)
    vrho = jnp.where(jnp.abs(rho0) < eps, 0.0, vrho)
    vsigma = jnp.where(jnp.abs(rho0) < eps, 0.0, vsigma)
    exc = np.asarray(exc)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)
    vxc = (vrho, vsigma, None, None)
    fxc = None
    kxc = None
    return exc, vxc, fxc, kxc


def eval_xc_GGA_pol_sara_STABLE(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                         custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.
    It takes into account polarization, but we do not calculate the hessian
    """
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    rho0u, rho0d, sigma1, sigma2, sigma3 = get_rho_sigma_polarized(rho)
    rho0 = jnp.array(rho0u+rho0d)
    sigma = sigma1+sigma2+sigma3
    sigma = jnp.array(sigma)

    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)
    def rho_times_func(rho0u, rho0d, sigma1, sigma2, sigma3):
        rho0 = jnp.array(rho0u+rho0d)
        sigma = sigma1+sigma2+sigma3
        sigma = jnp.array(sigma)
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrhou, vrhod, vsigma1, vsigma2, vsigma3 = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1, 2, 3, 4)))(rho0u, rho0d, sigma1, sigma2, sigma3)
    
    eps = 1e-14
    # IN the places where rho0 is smaller than eps, set exc to 0
    exc = jnp.where(jnp.abs(rho0) < eps, 0.0, exc)

    # If rho0u/rho0d is smaller than eps, set vrhou/vrhod to 0
    vrhou = jnp.where(jnp.abs(rho0u) < eps, 0.0, vrhou)
    vrhod = jnp.where(jnp.abs(rho0d) < eps, 0.0, vrhod)

    # For sigma, we set:
    def safe_vsigma(vsigma, rho_u, rho_d):
        # Sigma1 (σ↑↑) should be zero if ρ↑ is zero
        # Sigma3 (σ↓↓) should be zero if ρ↓ is zero
        # Sigma2 (σ↑↓) should be zero if either ρ↑ or ρ↓ is zero
        mask_up = rho_u < eps
        mask_dn = rho_d < eps

        # vsigma = (σ↑↑, σ↓↓, σ↑↓)
        vsigma_upup  = jnp.where(mask_up, 0.0, vsigma[0])
        vsigma_dndn  = jnp.where(mask_dn, 0.0, vsigma[2])
        vsigma_updn  = jnp.where(mask_up | mask_dn, 0.0, vsigma[1])
        return jnp.vstack([vsigma_upup, vsigma_dndn, vsigma_updn])

    vrho = jnp.vstack((vrhou, vrhod))
    vsigma = (vsigma1, vsigma2, vsigma3)
    vsigma = safe_vsigma(vsigma, rho0u, rho0d)
    vxc = (vrho, vsigma)
    fxc = None
    kxc = None
    TRANSPOSE = True
    if TRANSPOSE:
        vxc = [i.T for i in vxc]
    return exc, vxc, fxc, kxc