import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)  # Enables 64 bit precision


from functools import partial
from pyscf import dft
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import equinox as eqx


## COMMON FUNCTIONS - CUSTOM POTENTIALS
def custom_potential_ft_ex_sc(rho0, sigma, model_fx):
    """
    ft = from trained
    """
    grad_rho = jnp.sqrt(sigma)
    inputs = jnp.array([rho0, grad_rho])

    # Evaluate our model
    fx = model_fx(inputs)

    return fx*eUEG_LDA_x(rho0)

def custom_potential_ft_ec_sc(rho0, sigma, model_fc):
    """
    ft = from trained
    """
    grad_rho = jnp.sqrt(sigma)
    inputs = jnp.array([rho0, grad_rho])

    # Evaluate our model
    fc = model_fc(inputs)

    return fc*eUEG_LDA_c(rho0)

def custom_potential_ft_ex_sc_SIG(rho0, sigma, model_fx):
    """
    ft = from trained
    SIG = network expects sigma as input
    """
    inputs = jnp.array([rho0, sigma])

    # Evaluate our model
    fx = model_fx(inputs)

    return fx*eUEG_LDA_x(rho0)

def custom_potential_ft_ec_sc_SIG(rho0, sigma, model_fc):
    """
    ft = from trained
    SIG = network expects sigma as input
    """
    inputs = jnp.array([rho0, sigma])

    # Evaluate our model
    fc = model_fc(inputs)

    return fc*eUEG_LDA_c(rho0)

def custom_potential_ft_ex_sc_s(rho0, sigma, model_fx):
    """
    ft = from trained
    s = network expects s as input
    """
    k_F = (3 * jnp.pi**2 * rho0)**(1/3)
    s = jnp.sqrt(sigma) / (2 * k_F * rho0)
    inputs = jnp.array([rho0, s])
    # Evaluate our model
    fx = model_fx(inputs)

    return fx*eUEG_LDA_x(rho0)


def custom_potential_ft_ec_sc_s(rho0, sigma, model_fc):
    """
    ft = from trained
    s = network expects s as input
    """
    k_F = (3 * jnp.pi**2 * rho0)**(1/3)
    s = jnp.sqrt(sigma) / (2 * k_F * rho0)
    inputs = jnp.array([rho0, s])

    # Evaluate our model
    fc = model_fc(inputs)

    return fc*eUEG_LDA_c(rho0)


def eUEG_LDA_c(rho):
    """Perdew-Wang 1992 correlation energy density for unpolarized case.
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


def eUEG_LDA_x(rho):
    return -3/4*(3/jnp.pi)**(1/3)*jnp.sign(rho) * (jnp.abs(rho)) ** (1 / 3)

# General function to get the eval_xc function

def get_my_eval_xc_ft(model_fx, model_fc, sigma=False, polarized=True):
    """Get the xc functional to be used in PySCF.

    This function returns a function that can be used in PySCF, by using
    mf.define_xc_(my_eval_xc_ft, 'GGA').

    First, we define the custom ex/ec potentials used, which take rho and sigma
    as input  and return e as output. Different functions are used depending on whether
    the model expects sigma as input or grad rho. Summing up, the custom_ex/ec functions
    are defined as:
        custom_ex/ec(rho, sigma) = Fx/c(rho, grho or sigma) * eUEG_LDA_x/c(rho),
    where Fx/c is the output of the neural network. The definition of differents
    is to wrap my Fx/Fc in the correct way.

    If sigma is True, the model expects sigma as input.

    If polarized is True, the functional can also be used in spin-polarized
    calculations.
    """
    jax.config.update("jax_enable_x64", True)  # Enable 64 bit precision
    if sigma:
        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc_SIG,
                                     model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc_SIG,
                                     model_fc=model_fc)
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

        my_custom_pot_ft_x = partial(custom_potential_ft_ex_sc,
                                     model_fx=model_fx)
        my_custom_pot_ft_c = partial(custom_potential_ft_ec_sc,
                                     model_fc=model_fc)
    if polarized:
        my_eval_xc_ft = partial(eval_xc_GGA_pol_general, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    else:
        my_eval_xc_ft = partial(eval_xc_PBE_general, custom_ex=my_custom_pot_ft_x,
                                custom_ec=my_custom_pot_ft_c)
    return my_eval_xc_ft


def eval_xc_GGA_pol_general(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                            custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.

    It takes into account polarization, but we do not calculate the hessian"""
    # we only expect there to be a rho0 array, but I unpack it as (rho, deriv) here to be in line with the
    # pyscf example -- the size of the 'rho' array depends on the xc type (LDA, GGA, etc.)
    # so since LDA calculation, check for size first.
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    try:
        rhoshape = len(rho.shape)
        pol = 3
    except:
        rhoshape = len(rho)
        pol = 2
    # if len of shape == 3, spin polarized so compress to unpolarized for calculation
    if rhoshape != pol:
        # SPIN-UNPOLARIZED, ALL ARRAYS PASSED AS IS TO LIBXC
        # Same output as eval_xc_PBE_general
        exc, vxc, fxc, kxc = eval_xc_PBE_general(xc_code, rho, spin, relativity, deriv, omega, verbose, custom_ex=custom_ex, custom_ec=custom_ec)
    else:
        # SPIN POLARIZED; RESULT ARRAYS MUST BE RETURNED SPIN POLARIZED
        # THIS IS HACKY -- THE NETWORK IS NOT ARCHITECTED TO ACCEPT ALL THE POLARIZED PARAMETERS, SO THE GRADIENTS ARE JUST DUPLICATED IN THE RETURN;
        # GENERATE A FUNCTION THAT COMBINES THEN CALLS
        exc, vxc, fxc, kxc = eval_xc_GGA_pol_sara(xc_code, rho, spin, relativity, deriv, omega, verbose,
                                                  custom_ex=custom_ex, custom_ec=custom_ec)
    return exc, vxc, fxc, kxc


def eval_xc_PBE_general(xc_code, rho, spin=0, relativity=0, deriv=1,
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
    # Calculate the "custom" energy with rho -- THIS IS e
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)

    def rho_times_func(rho0, sigma):
        return rho0*custom_ex(rho0, sigma) + rho0*custom_ec(rho0, sigma)
    vrho, vsigma = jax.vmap(
        jax.grad(rho_times_func, argnums=(0, 1)))(rho0, sigma)

    # Sanity check, as in pyscf
    # Must go before converting to numpy
    eps = 1e-14
    exc = jnp.where(jnp.abs(rho0) < eps, 0.0, exc)
    vrho = jnp.where(jnp.abs(rho0) < eps, 0.0, vrho)
    vsigma = jnp.where(jnp.abs(rho0) < eps, 0.0, vsigma)

    # The outputs are expected to be numpy arrays
    exc = np.asarray(exc)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)
    vxc = (vrho, vsigma, None, None)
    fxc = None
    kxc = None

    return exc, vxc, fxc, kxc


def get_rho_sigma_polarized(rho):
    rho_u, rho_d = rho
    rho0u, dxu, dyu, dzu = rho_u[:4]
    rho0d, dxd, dyd, dzd = rho_d[:4]
    # up-up
    dxu2 = dxu*dxu
    dyu2 = dyu*dyu
    dzu2 = dzu*dzu
    # up-down
    dxud = dxu*dxd
    dyud = dyu*dyd
    dzud = dzu*dzd
    # down-down
    dxd2 = dxd*dxd
    dyd2 = dyd*dyd
    dzd2 = dzd*dzd
    sigma1 = dxu2+dyu2+dzu2
    sigma2 = dxud+dyud+dzud
    sigma3 = dxd2+dyd2+dzd2
    return rho0u, rho0d, sigma1, sigma2, sigma3


def eval_xc_GGA_pol_sara(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                         custom_ex=None, custom_ec=None):
    """Evaluate the GGA exchange-correlation functional.
    It takes into account polarization, but we do not calculate the hessian
    
    The definitions are:
    rho0 = rho_up + rho_down
    sigma1 = |grad rho_up|^2
    sigma2 = grad rho_up . grad rho_down
    sigma3 = |grad rho_down|^2
    sigma = sigma1 + sigma2 + sigma3
    """
    if custom_ex is None or custom_ec is None:
        raise ValueError("A functional must be specified!")
    # THIS IS HACKY -- THE NETWORK IS NOT ARCHITECTED TO ACCEPT ALL THE POLARIZED PARAMETERS, SO THE GRADIENTS ARE JUST DUPLICATED IN THE RETURN;
    # GENERATE A FUNCTION THAT COMBINES THEN CALLS

    rho0u, rho0d, sigma1, sigma2, sigma3 = get_rho_sigma_polarized(rho)
    rho0 = jnp.array(rho0u+rho0d)
    sigma = sigma1+sigma2+sigma3
    sigma = jnp.array(sigma)

    # calculate the "custom" energy with rho -- THIS IS e
    exc = jax.vmap(custom_ex)(rho0, sigma) + jax.vmap(custom_ec)(rho0, sigma)

    def rho_times_func(rho0u, rho0d, sigma1, sigma2, sigma3):
        # We need the derivative agains all the inputs
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
        vsigma_upup = jnp.where(mask_up, 0.0, vsigma[0])
        vsigma_dndn = jnp.where(mask_dn, 0.0, vsigma[2])
        vsigma_updn = jnp.where(mask_up | mask_dn, 0.0, vsigma[1])
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




# AS ALEC
def eval_xc_gga_pol(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                    xcmodel=None):
    # we only expect there to be a rho0 array, but I unpack it as (rho, deriv) here to be in line with the
    # pyscf example -- the size of the 'rho' array depends on the xc type (LDA, GGA, etc.)
    # so since LDA calculation, check for size first.
    try:
        rhoshape = len(rho.shape)
        pol = 3
    except:
        rhoshape = len(rho)
        pol = 2
    # if len of shape == 3, spin polarized so compress to unpolarized for calculation
    if rhoshape != pol:
        # SPIN-UNPOLARIZED, ALL ARRAYS PASSED AS IS TO LIBXC
        try:
            rho0, dx, dy, dz = rho[:4]
            sigma = jnp.array(dx**2+dy**2+dz**2)
        except:
            print("Unpacking failed...")
            rho0, drho = rho[:4]
            sigma = jnp.array(drho**2)
        rho0 = jnp.array(rho0)
        rhosig = jnp.stack([rho0, sigma], axis=1)
        # print('rho/sig/rhosig shapes: ', rho0.shape, sigma.shape, rhosig.shape)
        # calculate the "custom" energy with rho -- THIS IS e
        # cast back to np.array since that's what pyscf works with
        # pass as tuple -- (rho, sigma)
        exc = jax.vmap(xcmodel)(rhosig)
        exc = jnp.array(exc)/rho0
        vrho_f = eqx.filter_grad(xcmodel)
        vrhosigma = jnp.array(jax.vmap(vrho_f)(rhosig))
        vxc = (vrhosigma[:, 0], vrhosigma[:, 1], None, None)

        v2_f = jax.hessian(xcmodel)
        v2 = jnp.array(jax.vmap(v2_f)(rhosig))
        v2rho2 = v2[:, 0, 0]
        v2rhosigma = v2[:, 0, 1]
        v2sigma2 = v2[:, 1, 1]
        v2lapl2 = None
        vtau2 = None
        v2rholapl = None
        v2rhotau = None
        v2lapltau = None
        v2sigmalapl = None
        v2sigmatau = None
        # 2nd order functional derivative
        fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
        # 3rd order
        kxc = None

    else:
        # SPIN POLARIZED; RESULT ARRAYS MUST BE RETURNED SPIN POLARIZED
        # THIS IS HACKY -- THE NETWORK IS NOT ARCHITECTED TO ACCEPT ALL THE POLARIZED PARAMETERS, SO THE GRADIENTS ARE JUST DUPLICATED IN THE RETURN;
        # GENERATE A FUNCTION THAT COMBINES THEN CALLS
        def make_epsilon_function(model):
            # importantly, do not place the vmap here
            def get_epsilon(arr):
                rhou, rhod, sigma1, sigma2, sigma3 = arr
                rho0 = jnp.array(rhou+rhod)
                # sum the sigma contributions
                sumsigma = sigma1+sigma2+sigma3

                rhosig = jnp.stack([rho0, sumsigma])
                # calculate the "custom" energy with rho -- THIS IS e
                # cast back to np.array since that's what pyscf works with
                # pass as tuple -- (rho, sigma)
                exc = model(rhosig)
                return exc
            return get_epsilon

        # model_epsilon = partial(get_epsilon, model=xcmodel)
        model_epsilon = make_epsilon_function(model=xcmodel)
        rho_u, rho_d = rho
        # print('rho_u, rho_d shapes:', rho_u.shape, rho_d.shape)
        rho0u, dxu, dyu, dzu = rho_u[:4]
        rho0d, dxd, dyd, dzd = rho_d[:4]
        # up-up
        dxu2 = dxu*dxu
        dyu2 = dyu*dyu
        dzu2 = dzu*dzu
        # up-down
        dxud = dxu*dxd
        dyud = dyu*dyd
        dzud = dzu*dzd
        # down-down
        dxd2 = dxd*dxd
        dyd2 = dyd*dyd
        dzd2 = dzd*dzd
        sigma1 = dxu2+dyu2+dzu2
        sigma2 = dxud+dyud+dzud
        sigma3 = dxd2+dyd2+dzd2

        rho0 = jnp.array(rho0u+rho0d)
        # print('rho0 shape', rho0.shape)
        # print('sigma1/2/3 shapes', sigma1.shape, sigma2.shape, sigma3.shape)
        sumsigma = sigma1+sigma2+sigma3
        # print('sumsigma shape', sumsigma.shape)
        # sum the sigma contributions
        rhosig = jnp.stack([rho0, sigma1+sigma2+sigma3], axis=1)
        # calculate the "custom" energy with rho -- THIS IS e
        # cast back to np.array since that's what pyscf works with
        # pass as tuple -- (rho, sigma)
        # epsilon here
        input_arr = jnp.stack([rho0u, rho0d, sigma1, sigma2, sigma3], axis=1)
        exc = jax.vmap(model_epsilon)(input_arr)
        # print('epsilon shape', exc.shape)
        # e here
        exc = jnp.array(exc)/rho0
        # exc = exc[jnp.newaxis, :]
        # print('exc shape', exc.shape)
        v1_f = jax.grad(model_epsilon)
        v1 = jax.vmap(v1_f)(input_arr)
        # vrho = vrho_up, vrho_down
        vrho = jnp.vstack((v1[:, 0], v1[:, 1]))
        # vsigma = vsigma1, vsigma2, vsigma3
        vsigma = jnp.vstack((v1[:, 2], v1[:, 3], v1[:, 4]))
        vxc = (vrho, vsigma)
        # print('vrho shape', vrho.shape)
        # print('vsigma shape', vsigma.shape)
        v2_f = jax.hessian(model_epsilon)
        v2 = jax.vmap(v2_f)(input_arr)
        # print('v2 shape', v2.shape)
        # v2rho2 = (v2rhou2, v2rhoud, v2rhod2)
        v2rho2 = jnp.vstack((v2[:, 0, 0], v2[:, 0, 1], v2[:, 1, 1]))
        # v2rhosigma is six-part = (u,1),(u,2),(u,3),(d,1),(d,2),(d,3)
        v2rhosigma = jnp.vstack((v2[:, 0, 2], v2[:, 0, 3], v2[:, 0, 4], v2[:, 1, 2], v2[:, 1, 3], v2[:, 1, 4]))
        # v2sigma2 is also six-part
        v2sigma2 = jnp.vstack((v2[:, 2, 2], v2[:, 2, 3], v2[:, 2, 4], v2[:, 3, 3], v2[:, 3, 4], v2[:, 4, 4]))
        # print('v2rho2 shape', v2rho2.shape)
        # print('v2rhosigma shape', v2rhosigma.shape)
        # print('v2sigma2 shape', v2sigma2.shape)
        v2lapl2 = None
        vtau2 = None
        v2rholapl = None
        v2rhotau = None
        v2lapltau = None
        v2sigmalapl = None
        v2sigmatau = None
        # 2nd order functional derivative
        fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
        # 3rd order
        kxc = None
        TRANSPOSE = True
        if TRANSPOSE:
            vxc = [i.T for i in vxc]
            fxc = [i.T for i in fxc if type(i) == type(jnp.array([1]))]

    return exc, vxc, fxc, kxc



# -------------------- POST-PROCESSING FUNCTIONS ---------------------

def __get_energies_sfc(mol, eval_xc, max_steps=25):
    scf_steps = np.arange(0, max_steps+1)
    dict_energies = {step: {} for step in scf_steps}
    for step in scf_steps:
        this_mf = dft.RKS(mol)
        this_mf = this_mf.define_xc_(eval_xc, 'GGA')
        this_mf.max_cycle = step
        this_mf.kernel()
        this_dct = this_mf.scf_summary
        this_dct['e_tot'] = this_mf.e_tot
        this_dct['veff'] = this_mf.get_veff()
        dict_energies[step] = this_dct
    return dict_energies


def __get_errors_scf(dict_energies, mf_ref):
    scf_steps = dict_energies.keys()
    dict_errors = {step: {} for step in scf_steps}
    for step in scf_steps:
        this_dct = {}
        ref_dct = dict_energies[step]
        for key in ref_dct.keys():
            if key not in ['e_tot', 'veff']:
                this_dct[key] = np.abs(ref_dct[key]-mf_ref.scf_summary[key])
        this_dct['e_tot'] = np.abs(ref_dct['e_tot'] - mf_ref.e_tot)
        this_dct['Mean(veff)'] = np.mean(np.abs(ref_dct['veff']
                                                - mf_ref.get_veff()))
        dict_errors[step] = this_dct
    return dict_errors


def __errors_to_arrays(dict_errors):
    scf_steps = dict_errors.keys()
    energy_keys = dict_errors[0].keys()
    dict_arrays = {key: np.array([dict_errors[step][key]
                                  for step in scf_steps])
                   for key in energy_keys}
    return dict_arrays


def get_error_in_energies_sfc(mol, eval_xc, mf_ref, max_steps=25):
    """Get the errors in the energies and veff

    This error is computed by comparing the energies and veff of the
    calculations done with different number of SCF steps. The reference
    calculation is done with mf_ref, and the calculations with different
    number of SCF steps are done with the (custom) functional eval_xc, for the
    molecule 'mol'. The maximum number of SCF steps is max_steps.

    The form of the dictionary is:
        dict_arrays[energy_key] = np.array([error_step_0, error_step_1, ...])
     TODO
    """
    dict_energies = __get_energies_sfc(mol, eval_xc, max_steps)
    dict_errors = __get_errors_scf(dict_energies, mf_ref)
    dict_arrays = __errors_to_arrays(dict_errors)
    return dict_arrays


def plot_errors_pyscf(coords, ref, ft, lims, title):
    colorbar_positions = [0.25, 0.6, 0.95]
    # Create subplot figure (1 row, 3 columns)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"},
                {"type": "scatter3d"}]],
        horizontal_spacing=0.05
    )


    # Add three scatter plots
    for i in range(3):
        errors = np.abs(ref - ft)
        mask = errors > lims[i]
        coords_to_plot = coords[mask]
        charges = errors[mask]
        print('Number of points with error >', lims[i], ':', len(charges))
        fig.add_trace(
            go.Scatter3d(
                x=coords_to_plot[:, 0], y=coords_to_plot[:, 1], z=coords_to_plot[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=charges,  # Color by charge values
                    colorscale='Viridis',  # Different colormap per subplot
                    colorbar=dict(title=f"Error with lim {lims[i]}",
                                x=colorbar_positions[i],  # Move colorbar horizontally
                                len=0.75 ),
                    opacity=0.8
                )
            ),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Scatter3d(x=[0.0, 0.0, 0.0], y=[0.0, -0.757, 0.757], z=[0.0, 0.587, 0.587], mode='markers', marker=dict(size=12, color='red')),
            row=1, col=i+1
        )

    # Update layout
    fig.update_layout(
        title_text=f'Errors for {title}',
        showlegend=False,
        height=500, width=1200
    )
    fig.show()
