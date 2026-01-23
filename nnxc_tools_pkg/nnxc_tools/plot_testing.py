import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial
from modules.reference_functionals import Fc, Fx, FcSIG, FxSIG, \
    get_true_values_FC_grid, get_true_values_FX_grid, S, S_general, \
    Fc_s, Fx_s, \
    get_true_values_FX_grid_general, get_true_values_FC_grid_general
from modules.training_functions import compute_derivatives

# typing
import jax.typing as jt
from typing import Callable, Union


def calculate_stats(true: jt.ArrayLike, pred: jt.ArrayLike) -> \
        tuple:
    """Calculate statistics of the model

    Compares the array "true" and "pred" and gives:
    - R-squared value
    - Mean Absolute Error
    - Root Mean Squared Error

    To be used to compute the statistics between the output
    of a NN and the expected values.
    Args:
        true (jt.ArrayLike): True values
        pred (jt.ArrayLike): Predicted values

    Returns:
        tuple: A tuple containing:
            - r2 (float): R-squared value
            - mae (float): Mean Absolute Error
            - rmse (float): Root Mean Squared Error
    """
    # R-squared
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Mean Absolute Error
    mae = np.mean(np.abs(true - pred))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((true - pred) ** 2))

    return r2, mae, rmse


def plot_testinf_fxc(model_fx: Callable[[jt.ArrayLike, jt.ArrayLike],
                                        jt.ArrayLike],
                     model_fc: Callable[[jt.ArrayLike, jt.ArrayLike],
                                        jt.ArrayLike],
                     start_s: float = 0.0, stop_s: float = 5.0,
                     rho: float = 1.0, plot_all: bool = True):
    """
    Plots the testing results for the given models `model_fx` and `model_fc`.

    It only plots the results for the VALUES, without derivatives.

    Plots (for both Fx and Fc):
     1. True F vs Pred F
     2. True and Predicted values as a function of test s values.
     3. Absolute errors (difference between True and Predicted) as a function
     of the test s values.

    The values are ploted in the range [start_s, stop_s), with constant rho.

    Args:
        model_fx (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        model_fc (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the correlation enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0.
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.
        rho (float, optional): constant rho value. Defaults to 1.0.

    Returns:
        None
    """
    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fx_test = Fx(test_rho, grad_rho_test)
    true_fc_test = Fc(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fx = jax.vmap(model_fx)(test_inputs).squeeze()

    # The inputs here
    pred_fc = jax.vmap(model_fc)(test_inputs).squeeze()

    # Calculate statistics
    r2_fx, mae_fx, rmse_fx = calculate_stats(true_fx_test, pred_fx)
    r2_fc, mae_fc, rmse_fc = calculate_stats(true_fc_test, pred_fc)

    # 1. Correlation plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.scatter(true_fx_test, pred_fx, alpha=0.5)
    ax1.plot([true_fx_test.min(), true_fx_test.max()],
             [true_fx_test.min(), true_fx_test.max()], 'r--')
    ax1.set_xlabel('True Fx')
    ax1.set_ylabel('Predicted Fx')
    ax1.set_title(f'''True vs Predicted Fx\nR² = {r2_fx:.4f},
MAE = {mae_fx:.4f}, RMSE = {rmse_fx:.4f}''')

    ax2.scatter(true_fc_test, pred_fc, alpha=0.5)
    ax2.plot([true_fc_test.min(), true_fc_test.max()],
             [true_fc_test.min(), true_fc_test.max()], 'r--')
    ax2.set_xlabel('True Fc')
    ax2.set_ylabel('Predicted Fc')
    ax2.set_title(f'''True vs Predicted Fc\nR² = {r2_fc:.4f},
MAE = {mae_fc:.4f}, RMSE = {rmse_fc:.4f}''')

    plt.tight_layout()
    plt.show()

    # 2. True PBE factors vs. s
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.plot(test_s, true_fx_test, label='True Fx')
    ax1.plot(test_s, pred_fx, label='Predicted Fx')
    ax1.set_xlabel('s')
    ax1.set_ylabel('Fx')
    ax1.set_title(f'''Fx vs s\nR² = {r2_fx:.4f},
MAE = {mae_fx:.4f}, RMSE = {rmse_fx:.4f}''')
    ax1.legend()

    ax2.plot(test_s, true_fc_test, label='True Fx')
    ax2.plot(test_s, pred_fc, label='Predicted Fx')
    ax2.set_xlabel('s')
    ax2.set_ylabel('Fc')
    ax2.set_title(f'''Fc vs s\nR² = {r2_fc:.4f},
MAE = {mae_fc:.4f}, RMSE = {rmse_fc:.4f}''')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # 3. Error plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.plot(test_s, np.abs(true_fx_test - pred_fx))
    ax1.set_xlabel('s')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Absolute Error for Fx vs s')

    ax2.plot(test_s, np.abs(true_fc_test - pred_fc))
    ax2.set_xlabel('s')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Error for Fc vs s')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("Fx Statistics:")
    print(f"R-squared: {r2_fx:.6f}")
    print(f"Mean Absolute Error: {mae_fx:.6f}")
    print(f"Root Mean Squared Error: {rmse_fx:.6f}")
    print(f"""Maximum Absolute Error:
          {np.max(np.abs(true_fx_test - pred_fx)):.6f}""")

    print("\nFc Statistics:")
    print(f"R-squared: {r2_fc:.6f}")
    print(f"Mean Absolute Error: {mae_fc:.6f}")
    print(f"Root Mean Squared Error: {rmse_fc:.6f}")
    print(f"""Maximum Absolute Error:
          {np.max(np.abs(true_fc_test - pred_fc)):.6f}""")


def plot_testinf_fc_s(model_fc: Callable[[jt.ArrayLike, jt.ArrayLike],
                                         jt.ArrayLike],
                      start_s: float = 0.0, stop_s: float = 5.0,
                      save: bool = False, figname: str = 'fcs.png',
                      title: Union[str, None] = None, rho: float = 1.0):
    """Plot Fc vs s with gradients.

    Fc must expect as input: (rho, grad_rho).

    Args:
        model_fc (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the correlation enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fcs.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        rho (float, optional): constant rho value. Defaults to 1.0

    Returns:
        None
    """

    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fc_test = Fc(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fc = jax.vmap(model_fc)(test_inputs).squeeze()

    def scalar_Fc(rho, grad_rho):
        return Fc(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
    true_fc_grad_rho, true_fc_grad_grad_rho = \
        grad_Fc(test_rho.flatten(), grad_rho_test.flatten())
    pred_grad_rho_fc, pred_grad_grad_rho_fc = \
        jax.vmap(partial(compute_derivatives, model_fc))(test_inputs)

    __do_derivatives_plot(test_s, true_fc_test, pred_fc,
                          true_fc_grad_rho, pred_grad_rho_fc,
                          true_fc_grad_grad_rho, pred_grad_grad_rho_fc,
                          save=save, figname=figname, title=title, model='Fc',
                          xlabel='s')


def plot_testinf_fx_s(model_fx: Callable[[jt.ArrayLike, jt.ArrayLike],
                                         jt.ArrayLike],
                      start_s: float = 0.0, stop_s: float = 5.0,
                      save: bool = False, figname: str = 'fxs.png',
                      title: Union[str, None] = None, rho: float = 1.0):
    """Plot Fx vs s with gradients.

    Fx must expect as input: (rho, grad_rho).
    Args:
        model_fx (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fxs.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        rho (float, optional): constant rho value. Defaults to 1.0

    Returns:
        None
    """
    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fx_test = Fx(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fx = jax.vmap(model_fx)(test_inputs).squeeze()

    def scalar_Fx(rho, grad_rho):
        return Fx(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
    true_fx_grad_rho, true_fx_grad_grad_rho = \
        grad_Fx(test_rho.flatten(), grad_rho_test.flatten())
    pred_grad_rho_fx, pred_grad_grad_rho_fx = \
        jax.vmap(partial(compute_derivatives, model_fx))(test_inputs)

    __do_derivatives_plot(test_s, true_fx_test, pred_fx,
                          true_fx_grad_rho, pred_grad_rho_fx,
                          true_fx_grad_grad_rho, pred_grad_grad_rho_fx,
                          save=save, figname=figname, title=title, model='Fx',
                          xlabel='s')


def plot_testinf_fxc_s(model_fx: Callable[[jt.ArrayLike, jt.ArrayLike],
                                          jt.ArrayLike],
                       model_fc: Callable[[jt.ArrayLike, jt.ArrayLike],
                                          jt.ArrayLike],
                       start_s: float = 0.0, stop_s: float = 5.0,
                       save: bool = False,
                       figname: list[str] = ['fxs.png', 'fcs.png'],
                       title: Union[list[str], None] = None, rho: float = 1.0,
                       sinput: bool = False):
    """Plot Fx and Fc vs s with gradients.

    Args:
        model_fx (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        model_fc (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the correlation enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (list[str], optional): In case of saving the images, the name
            of these. The first name in the list is for Fx, the second for Fc.
            Defaults to ['fxs.png', 'fcs.png']
        title (Union[list[str], None], optional): Title to put to the plot.
            If None, no title. Defaults to None
        rho (float, optional): constant rho value. Defaults to 1.0
        sinput (bool, optional): If True, the input of the model is s, not
            grad_rho. Defaults to False.

    Returns:
        None
    """
    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    if sinput:
        test_inputs = jnp.stack([test_rho, test_s], axis=1)
    else:
        test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fc_test = Fc(test_rho, grad_rho_test)
    true_fx_test = Fx(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fc = jax.vmap(model_fc)(test_inputs).squeeze()
    pred_fx = jax.vmap(model_fx)(test_inputs).squeeze()

    if sinput:
        def scalar_Fc_s(rho, s):
            return Fc_s(jnp.array([rho]), jnp.array([s]))[0]

        def scalar_Fx_s(rho, s):
            return Fx_s(jnp.array([s]))[0]
        grad_Fc = jax.vmap(jax.grad(scalar_Fc_s, argnums=(0, 1)))
        true_fc_grad_rho, true_fc_grad_grad_rho = \
            grad_Fc(test_rho.flatten(), test_s.flatten())
        grad_Fx = jax.vmap(jax.grad(scalar_Fx_s, argnums=(0, 1)))
        true_fx_grad_rho, true_fx_grad_grad_rho = \
            grad_Fx(test_rho.flatten(), test_s.flatten())
    else:
        def scalar_Fc(rho, grad_rho):
            return Fc(jnp.array([rho]), jnp.array([grad_rho]))[0]

        def scalar_Fx(rho, grad_rho):
            return Fx(jnp.array([rho]), jnp.array([grad_rho]))[0]
        grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
        true_fc_grad_rho, true_fc_grad_grad_rho = \
            grad_Fc(test_rho.flatten(), grad_rho_test.flatten())
        grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
        true_fx_grad_rho, true_fx_grad_grad_rho = \
            grad_Fx(test_rho.flatten(), grad_rho_test.flatten())

    # Compute gradients
    pred_grad_rho_fc, pred_grad_grad_rho_fc = \
        jax.vmap(partial(compute_derivatives, model_fc))(test_inputs)

    pred_grad_rho_fx, pred_grad_grad_rho_fx = \
        jax.vmap(partial(compute_derivatives, model_fx))(test_inputs)

    if title is None:
        title = ['Fx', 'Fc']
    __do_derivatives_plot(test_s, true_fx_test, pred_fx,
                          true_fx_grad_rho, pred_grad_rho_fx,
                          true_fx_grad_grad_rho, pred_grad_grad_rho_fx,
                          save=save, figname=figname[0], title=title[0],
                          xlabel='s')
    __do_derivatives_plot(test_s, true_fc_test, pred_fc,
                          true_fc_grad_rho, pred_grad_rho_fc,
                          true_fc_grad_grad_rho, pred_grad_grad_rho_fc,
                          save=save, figname=figname[1], title=title[1],
                          xlabel='s')


def plot_testinf_fc_rho(model_fc: Callable[[jt.ArrayLike, jt.ArrayLike],
                                           jt.ArrayLike],
                        start_rho: float = 0.0, stop_rho: float = 1.0,
                        save: bool = False, figname: str = 'fcrho.png',
                        title: Union[str, None] = None, s: float = 0.1):
    """Plot Fc vs rho with gradients.

    Plots testing results for the correlation model (values and gradients wrt
    rho and ∇rho) as a function of density (rho).

    Args:
        model_fc (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained correlation
            model we wish to test and plot.
        start_rho (float, optional): Start of the interval for the test rho
            values. Defaults to 0.0
        stop_rho (float, optional): End of the interval for the test rho
            values.Defaults to 1.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fcrho.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        s (float, optional): Defaults to 0.1

    Returns:
        None
    """
    # Generate test points
    test_rho = jnp.linspace(start_rho, stop_rho, num=1000)
    test_s = jnp.full_like(test_rho, s)  # Constant s for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fc_test = Fc(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fc = jax.vmap(model_fc)(test_inputs).squeeze()

    def scalar_Fc(rho, grad_rho):
        return Fc(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
    true_fc_grad_rho, true_fc_grad_grad_rho = \
        grad_Fc(test_rho.flatten(), grad_rho_test.flatten())
    pred_grad_rho_fc, pred_grad_grad_rho_fc = \
        jax.vmap(partial(compute_derivatives, model_fc))(test_inputs)

    if title is None:
        title = 'Fc'

    __do_derivatives_plot(test_s, true_fc_test, pred_fc,
                          true_fc_grad_rho, pred_grad_rho_fc,
                          true_fc_grad_grad_rho, pred_grad_grad_rho_fc,
                          save=save, figname=figname, title=title,
                          xlabel='rho')


def plot_testinf_fx_rho(model_fx: Callable[[jt.ArrayLike,
                                           jt.ArrayLike], jt.ArrayLike],
                        start_rho: float = 0.0, stop_rho: float = 1.0,
                        save: bool = False, figname: str = 'fxrho.png',
                        title: Union[str, None] = None, s: float = 0.1):
    """Plot Fx vs rho with gradients.

    Plots testing results for the model (values and gradients with respect rho
    and ∇rho) as a function of density (rho).

    Args:
        model_fx (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        start_rho (float, optional): Start of the interval for the test rho
            values.Defaults to 0.0
        stop_rho (float, optional): End of the interval for the test rho
            values.Defaults to 1.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fcrho.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        s (float, optional): Defaults to 0.1

    Returns:
        None
    """
    # Generate test points
    test_rho = jnp.logspace(start_rho, stop_rho, num=1000)
    test_s = jnp.full_like(test_rho, s)  # Constant s for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    grad_rho_test = 2 * test_s * k_F_test * test_rho

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, grad_rho_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fx_test = Fx(test_rho, grad_rho_test)

    # Predict using neural networks
    pred_fx = jax.vmap(model_fx)(test_inputs).squeeze()

    def scalar_Fx(rho, grad_rho):
        return Fx(jnp.array([rho]), jnp.array([grad_rho]))[0]

    # Compute gradients
    grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
    true_fx_grad_rho, true_fx_grad_grad_rho = \
        grad_Fx(test_rho.flatten(), grad_rho_test.flatten())
    pred_grad_rho_fx, pred_grad_grad_rho_fx = \
        jax.vmap(partial(compute_derivatives, model_fx))(test_inputs)

    if title is None:
        title = 'Fx'

    __do_derivatives_plot(test_rho, true_fx_test, pred_fx,
                          true_fx_grad_rho, pred_grad_rho_fx,
                          true_fx_grad_grad_rho, pred_grad_grad_rho_fx,
                          save=save, figname=figname, title=title,
                          xlabel='rho')


def plot_testinf_fxSIG_s(model_fx: Callable[[jt.ArrayLike, jt.ArrayLike],
                                            jt.ArrayLike],
                         start_s: float = 1e-5, stop_s: float = 5.0,
                         save: bool = False, figname: str = 'fx_Sig.png',
                         title: Union[str, None] = None, rho: float = 1.0):
    """Plot Fx vs s with gradients.

    The input that the network expects are rho and sigma, so the derivatives
    are taken with respect to these two variables (creo).

    Args:
        model_fx (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fxs.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        rho (float, optional): constant rho value. Defaults to 1.0

    Returns:
        None
    """
    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    sigma_test = (2 * test_s * k_F_test * test_rho)**2

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, sigma_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fx_test = FxSIG(test_rho, sigma_test)

    # Predict using neural networks
    pred_fx = jax.vmap(model_fx)(test_inputs).squeeze()

    def scalar_Fx(rho, sigma):
        return FxSIG(jnp.array([rho]), jnp.array([sigma]))[0]

    # Compute gradients
    grad_Fx = jax.vmap(jax.grad(scalar_Fx, argnums=(0, 1)))
    true_fx_grad_rho, true_fx_grad_sigma = \
        grad_Fx(test_rho.flatten(), sigma_test.flatten())
    pred_grad_rho_fx, pred_grad_sigma_fx = \
        jax.vmap(partial(compute_derivatives, model_fx))(test_inputs)

    __do_derivatives_plot(test_s, true_fx_test, pred_fx,
                          true_fx_grad_rho, pred_grad_rho_fx,
                          true_fx_grad_sigma, pred_grad_sigma_fx,
                          save=save, figname=figname, title=title, model='Fx',
                          xlabel='s')


def plot_testinf_fcSIG_s(model_fc: Callable[[jt.ArrayLike, jt.ArrayLike],
                                            jt.ArrayLike],
                         start_s: float = 1e-5, stop_s: float = 5.0,
                         save: bool = False, figname: str = 'fc_Sig.png',
                         title: Union[str, None] = None, rho: float = 1.0):
    """Plot Fc vs s with gradients.

    The input that the network expects are rho and sigma, so the derivatives
    are taken with respect to these two variables (creo).

    Args:
        model_fc (Callable[[jt.ArrayLike, jt.ArrayLike]]): trained function for
            the exchange enhancement factor.
        start_s (float, optional): Start of the interval for the test s values.
            Defaults to 0.0
        stop_s (float, optional): End of the interval for the test s values.
            Defaults to 5.0
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'fcs.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        rho (float, optional): constant rho value. Defaults to 1.0

    Returns:
        None
    """
    # Generate test points
    test_s = jnp.linspace(start_s, stop_s, num=1000)
    test_rho = jnp.full_like(test_s, rho)  # Constant rho for simplicity

    # Calculate grad_rho for the test points
    k_F_test = (3 * jnp.pi**2 * test_rho)**(1/3)
    sigma_test = (2 * test_s * k_F_test * test_rho)**2

    # Prepare inputs for the neural networks
    test_inputs = jnp.stack([test_rho, sigma_test], axis=1)

    # Calculate true PBE enhancement factors
    true_fc_test = FcSIG(test_rho, sigma_test)

    # Predict using neural networks
    pred_fc = jax.vmap(model_fc)(test_inputs).squeeze()

    def scalar_Fc(rho, sigma):
        return FcSIG(jnp.array([rho]), jnp.array([sigma]))[0]

    # Compute gradients
    grad_Fc = jax.vmap(jax.grad(scalar_Fc, argnums=(0, 1)))
    true_fc_grad_rho, true_fc_grad_sigma = \
        grad_Fc(test_rho.flatten(), sigma_test.flatten())
    pred_grad_rho_fc, pred_grad_sigma_fc = \
        jax.vmap(partial(compute_derivatives, model_fc))(test_inputs)

    __do_derivatives_plot(test_s, true_fc_test, pred_fc,
                          true_fc_grad_rho, pred_grad_rho_fc,
                          true_fc_grad_sigma, pred_grad_sigma_fc,
                          save=save, figname=figname, title=title, model='Fc',
                          xlabel='s')


def __do_derivatives_plot(test_points: jt.ArrayLike, true_F: jt.ArrayLike,
                          pred_F: jt.ArrayLike,
                          true_F_grad_rho: jt.ArrayLike,
                          pred_F_grad_rho: jt.ArrayLike,
                          true_F_grad_grad_rho: jt.ArrayLike,
                          pred_F_grad_grad_rho: jt.ArrayLike,
                          save: bool = False, figname: str = 'dplot.png',
                          title: Union[str, None] = None, model: str = 'F',
                          xlabel: str = 'Unknown'):
    """
    Creates a plot of derivatives for a given model.

    Args:
        test_points (jt.ArrayLike): points where the model was tested.
        true_F (jt.ArrayLike): true values of the model (F).
        pred_F (jt.ArrayLike): predicted values of the model (F).
        true_F_grad_rho (jt.ArrayLike): true gradient of F wrt rho.
        pred_F_grad_rho (jt.ArrayLike): predicted gradient of F wrt rho.
        true_F_grad_grad_rho (jt.ArrayLike): true gradient of F wrt
            the gradient of rho.
        pred_F_grad_grad_rho (jt.ArrayLike): predicted gradient of F wrt
            the gradient of rho.
        save (bool, optional): Whether to save the output image.
            Defaults to False
        figname (str, optional): In case of saving the image, the name of this.
            Defaults to 'dplot.png'
        title (Union[str, None], optional): Title to put to the plot. If None,
            no title. Defaults to None
        model (str, optional): Which model we are plottig (serves to specify
            Fx or Fc to have the labels correctly). Defaults to 'F'
        xlabel (str, optional): label for the x-axis. It is used to specify
            if we are plotting agains rho or s. Defaults to 'Unknown'

    Returns:
        None
    """
    # Calculate statistics
    r2_fc, mae_fc, rmse_fc = calculate_stats(true_F, pred_F)
    r2_gfc, mae_gfc, rmse_gfc = calculate_stats(true_F_grad_rho,
                                                pred_F_grad_rho)
    r2_ggfc, mae_ggfc, rmse_ggfc = calculate_stats(true_F_grad_grad_rho,
                                                   pred_F_grad_grad_rho)
    m = model
    x = xlabel

    # 1. Correlation plots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    if title is not None:
        fig.suptitle(f'{title}')

    ax1, ax2, ax3 = axes[0, :]
    ax1.scatter(true_F, pred_F, alpha=0.5)
    ax1.plot([true_F.min(), true_F.max()],
             [true_F.min(), true_F.max()], 'r--')
    ax1.set_xlabel(f'True {m}')
    ax1.set_ylabel(f'Predicted {m}')
    ax1.set_title(f'''True vs Predicted {m}\nR² = {r2_fc:.4f},
    MAE = {mae_fc:.4f}, RMSE = {rmse_fc:.4f}''')

    ax2.scatter(true_F_grad_rho, pred_F_grad_rho, alpha=0.5)
    ax2.plot([true_F_grad_rho.min(), true_F_grad_rho.max()],
             [true_F_grad_rho.min(), true_F_grad_rho.max()], 'r--')
    ax2.set_xlabel(fr'True $\delta${m}/$\delta$n')
    ax2.set_ylabel(fr'Predicted $\delta${m}/$\delta$n')
    ax2.set_title(f'''True vs Predicted $\delta${m}/$\delta$n
                  R² = {r2_gfc:.4f},
    MAE = {mae_gfc:.4f}, RMSE = {rmse_gfc:.4f}''')

    ax3.scatter(true_F_grad_grad_rho, pred_F_grad_grad_rho, alpha=0.5)
    ax3.plot([true_F_grad_grad_rho.min(), true_F_grad_grad_rho.max()],
             [true_F_grad_grad_rho.min(), true_F_grad_grad_rho.max()], 'r--')
    ax3.set_xlabel(fr'True $\delta${m}/$\delta$($\nabla$n)')
    ax3.set_ylabel(fr'Predicted $\delta${m}/$\delta$($\nabla$n)')
    text = (fr'True vs Predicted $\delta${m}/$\delta\nabla$(n)'
            f'''\nR² = {r2_ggfc:.4f},
            MAE = {mae_ggfc:.4f}, RMSE = {rmse_ggfc:.4f}''')
    ax3.set_title(text)

    # 2. True PBE factors vs. xlabel (the one given)

    ax1, ax2, ax3 = axes[1, :]
    ax1.plot(test_points, true_F, label=f'True {m}')
    ax1.plot(test_points, pred_F, label=f'Predicted {m}')
    ax1.set_xlabel(f'{x}')
    ax1.set_ylabel(f'{m}')
    ax1.set_title(f'''{m} vs {x}\nR² = {r2_fc:.4f},
    MAE = {mae_fc:.4f}, RMSE = {rmse_fc:.4f}''')
    ax1.legend()

    ax2.plot(test_points, true_F_grad_rho,
             label=fr'True $\delta${m}/$\delta$n')
    ax2.plot(test_points, pred_F_grad_rho,
             label=fr'Predicted $\delta${m}/$\delta$n')
    ax2.set_xlabel(f'{x}')
    ax2.set_ylabel(fr'$\delta${m}/$\delta$n')
    ax2.set_title(f'''$\delta${m}/$\delta$n vs {x}\nR² = {r2_gfc:.4f},
    MAE = {mae_gfc:.4f}, RMSE = {rmse_gfc:.4f}''')
    ax2.legend()

    ax3.plot(test_points, true_F_grad_grad_rho,
             label=fr'True $\delta${m}/$\delta$($\nabla$n)')
    ax3.plot(test_points, pred_F_grad_grad_rho,
             label=fr'Predicted $\delta${m}/$\delta$($\nabla$n)')
    ax3.set_xlabel(f'{x}')
    ax3.set_ylabel(fr'$\delta${m}/$\delta$($\nabla$n)')
    text = (fr'$\delta${m}/$\delta$($\nabla$n) vs {x}' +
            f'''\nR² = {r2_ggfc:.4f},\n MAE = {mae_ggfc:.4f},
            RMSE = {rmse_ggfc:.4f}''')
    ax3.set_title(text)
    ax3.legend()

    # 3. Error plots
    ax1, ax2, ax3 = axes[2, :]

    ax1.plot(test_points, np.abs(true_F - pred_F))
    ax1.set_xlabel(f'{x}')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title(f'Absolute Error for {m} vs {x}')

    ax2.plot(test_points, np.abs(true_F_grad_rho - pred_F_grad_rho))
    ax2.set_xlabel(f'{x}')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(fr'Absolute Error for $\delta${m}/$\delta$n vs {x}')

    ax3.plot(test_points, np.abs(true_F_grad_grad_rho - pred_F_grad_grad_rho))
    ax3.set_xlabel(f'{x}')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title(fr'Abs. Error for $\delta${m}/$\delta$($\nabla$n) vs {x}')
    if x.lower() == 'rho':
        for i in [1, 2]:
            for j in range(3):
                axes[i, j].set_xscale('log')
    plt.tight_layout()
    if save:
        plt.savefig(figname, dpi=100)
    plt.show()

    # Print statistics
    print(f"\n{m} Statistics:")
    print(f"R-squared: {r2_fc:.6f}")
    print(f"Mean Absolute Error: {mae_fc:.6f}")
    print(f"Root Mean Squared Error: {rmse_fc:.6f}")
    print(f"""Maximum Absolute Error:
          {np.max(np.abs(true_F - pred_F)):.6f}""")


def plot_heatmap_fc(model, npts, rho_lims, s_lims, title=None):
    inputs, true_fc, _ = get_true_values_FC_grid(npts, s_lims=s_lims,
                                                 rho_lims=rho_lims)
    # inputs = sinputs
    pred_fc = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))
    true_fc = true_fc.reshape(npts_axis, npts_axis)
    pred_fc = pred_fc.reshape(npts_axis, npts_axis)
    # s_vals = S(inputs[:, 0], inputs[:, 1])
    # plot_heatmap_fv(true_fc, pred_fc, inputs[:, 0], s_vals,
    #                 model='Fc', title=title)
    plot_heatmap_fv(inputs, true_fc, pred_fc, 'Fc', title=title, 
                    input_var='grad_rho')

def plot_heatmap_fx(model, npts, rho_lims, s_lims, title=None):
    inputs, true_fx, _ = get_true_values_FX_grid(npts, s_lims=s_lims,
                                                 rho_lims=rho_lims)
    # inputs = sinputs
    pred_fx = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))
    true_fx = true_fx.reshape(npts_axis, npts_axis)
    pred_fx = pred_fx.reshape(npts_axis, npts_axis)
    # s_vals = S(inputs[:, 0], inputs[:, 1])
    # plot_heatmap_fv(true_fx, pred_fx, inputs[:, 0], s_vals,
    #                 model='Fx', title=title)
    plot_heatmap_fv(inputs, true_fx, pred_fx, 'Fx', title=title, 
                    input_var='grad_rho')


def plot_heatmap_fx_SIG(model, npts=10000, rho_lims=[0.01, 5],
                        s_lims=[0.01, 5], title=None):
    inputs, true_fx, _ = \
        get_true_values_FX_grid_general(npts, s_lims=s_lims, rho_lims=rho_lims,
                                        deriv_wrt='grad_rho',
                                        input_var='sigma')
    pred_fx = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))
    true_fx = true_fx.reshape(npts_axis, npts_axis)
    pred_fx = pred_fx.reshape(npts_axis, npts_axis)
    # s_vals = S(inputs[:, 0], np.sqrt(inputs[:, 1]))
    # inputs = sinputs
    plot_heatmap_fv(inputs, true_fx, pred_fx, 'Fx', title=title,
                    input_var='sigma')


def plot_heatmap_fc_SIG(model, npts=10000, rho_lims=[0.01, 5],
                        s_lims=[0.01, 5], title=None):
    inputs, true_fc, _ = \
        get_true_values_FC_grid_general(npts, s_lims=s_lims, rho_lims=rho_lims,
                                        deriv_wrt='grad_rho',
                                        input_var='sigma')
    pred_fc = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))
    true_fc = true_fc.reshape(npts_axis, npts_axis)
    pred_fc = pred_fc.reshape(npts_axis, npts_axis)
    # s_vals = S(inputs[:, 0], np.sqrt(inputs[:, 1]))
    # inputs = sinputs
    # plot_heatmap_fv(true_fc, pred_fc, inputs[:, 0], s_vals,
    #                 model='Fc', title=title)
    plot_heatmap_fv(inputs, true_fc, pred_fc, 'Fc', title=title,
                    input_var='sigma')


def plot_heatmap_fv(inputs, true_f, pred_f,  model_name,
                        title=None, input_var=None):
    npts_axis = int(jnp.sqrt(inputs.shape[0]))
    xv = inputs[:, 0].reshape(npts_axis, npts_axis)
    yv = S_general(inputs[:, 0], inputs[:, 1], inp_var=input_var).reshape(npts_axis, npts_axis)
    true_f = true_f.reshape(npts_axis, npts_axis)
    pred_f = pred_f.reshape(npts_axis, npts_axis)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

    
    vmax = max(jnp.max(true_f), jnp.max(pred_f))
    vmin = min(jnp.min(true_f), jnp.min(pred_f))

    if title is not None:
        fig.suptitle(f'{title}')
    im1 = ax1.pcolormesh(xv, yv, true_f, shading='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(f'True {model_name}')
    ax1.set_xlabel(r'$\rho$')
    ax1.set_xscale('log')
    ax1.set_ylabel('s')
    cbar1 = fig.colorbar(im1, ax=ax1)
    
    im2 = ax2.pcolormesh(xv, yv, pred_f, shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Predicted {model_name}')
    ax2.set_xlabel(r'$\rho$')
    ax2.set_xscale('log')
    ax2.set_ylabel('s')
    cbar2 = fig.colorbar(im2, ax=ax2)

    
    error_val = np.abs(true_f - pred_f)
    im3 = ax3.pcolormesh(xv, yv, error_val, shading='auto')
    ax3.set_title(f'Abs. Error for {model_name}')
    ax3.set_xlabel(r'$\rho$')
    ax3.set_xscale('log')
    ax3.set_ylabel('s')
    cbar3 = fig.colorbar(im3, ax=ax3)


    plt.tight_layout()
    plt.show()


def plot_heatmap_fv_old(true, pred, rho_values, s_values, model, title=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    if title is not None:
        fig.suptitle(f'{title}')
    vmax = max(jnp.max(true), jnp.max(pred))
    vmin = min(jnp.min(true), jnp.min(pred))
    im1 = ax1.imshow(true.T, aspect='auto', origin='lower',
                     extent=[s_values[0], s_values[-1],
                             rho_values[0], rho_values[-1]],
                             vmin=vmin, vmax=vmax)

    ax1.set_title(f'True {model}')
    ax1.set_xlabel('s')
    ax1.set_ylabel('rho')
    ax1.set_yscale('log')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(pred.T, aspect='auto', origin='lower',
                     extent=[s_values[0], s_values[-1],
                             rho_values[0], rho_values[-1]],
                             vmin=vmin, vmax=vmax)
    ax2.set_title(f'Predicted {model}')
    ax2.set_xlabel('s')
    ax2.set_ylabel('rho')
    ax2.set_yscale('log')
    fig.colorbar(im2, ax=ax2)

    error = np.abs(true - pred)
    im3 = ax3.imshow(error.T, aspect='auto', origin='lower',
                     extent=[s_values[0], s_values[-1],
                             rho_values[0], rho_values[-1]])
    ax3.set_title(f'Absolute Error for {model}')
    ax3.set_xlabel('s')
    ax3.set_ylabel('rho')
    ax3.set_yscale('log')
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.show()


def plot_heatmap_fx_g_gg(model, npts=10000,
                         rho_lims=[0.01, 5], s_lims=[0.01, 5], title=None,
                         short_model_name=True):
    inputs, true_fx, true_grads_fx = get_true_values_FX_grid(npts,
                                                             rho_lims,
                                                             s_lims)
    true_grad_fx, true_grad_grho_fx = true_grads_fx

    pred_fx = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))

    pred_grad_rho_fx, pred_grad_grho_fx = \
        jax.vmap(partial(compute_derivatives, model))(inputs)

    pred_fx = pred_fx.reshape(npts_axis, npts_axis)
    pred_grad_rho_fx = pred_grad_rho_fx.reshape(npts_axis, npts_axis)
    pred_grad_grho_fx = pred_grad_grho_fx.reshape(npts_axis, npts_axis)

    true_fx = true_fx.reshape(npts_axis, npts_axis)
    true_grad_fx = true_grad_fx.reshape(npts_axis, npts_axis)
    true_grad_grho_fx = true_grad_grho_fx.reshape(npts_axis, npts_axis)

    s_values = jnp.linspace(*s_lims, num=npts_axis)
    rho_values = jnp.logspace(jnp.log10(rho_lims[0]), jnp.log10(rho_lims[1]),
                              num=npts_axis)
    if short_model_name:
        model_name = model.name[0:2]
    else:
        model_name = model.name

    plot_heatmap_fv_g_gg(s_values, rho_values,
                         true_fx, pred_fx,
                         true_grad_fx, pred_grad_rho_fx,
                         true_grad_grho_fx, pred_grad_grho_fx,
                         model_name, title)


def plot_heatmap_fc_g_gg(model, npts=10000,
                         rho_lims=[0.01, 5], s_lims=[0.01, 5], title=None,
                         short_model_name=True):
    inputs, true_fc, true_grads_fc = get_true_values_FC_grid(npts, rho_lims,
                                                             s_lims)
    true_grad_fc, true_grad_grho_fc = true_grads_fc

    pred_fc = jax.vmap(model)(inputs).squeeze()
    npts_axis = int(jnp.sqrt(inputs.shape[0]))

    pred_grad_rho_fc, pred_grad_grho_fc = \
        jax.vmap(partial(compute_derivatives, model))(inputs)

    pred_fc = pred_fc.reshape(npts_axis, npts_axis)
    pred_grad_rho_fc = pred_grad_rho_fc.reshape(npts_axis, npts_axis)
    pred_grad_grho_fc = pred_grad_grho_fc.reshape(npts_axis, npts_axis)

    true_fc = true_fc.reshape(npts_axis, npts_axis)
    true_grad_fc = true_grad_fc.reshape(npts_axis, npts_axis)
    true_grad_grho_fc = true_grad_grho_fc.reshape(npts_axis, npts_axis)

    s_values = jnp.linspace(*s_lims, num=npts_axis)
    rho_values = jnp.logspace(jnp.log10(rho_lims[0]), jnp.log10(rho_lims[1]),
                              num=npts_axis)
    if short_model_name:
        model_name = model.name[0:2]
    else:
        model_name = model.name

    plot_heatmap_fv_g_gg(s_values, rho_values, true_fc, pred_fc,
                         true_grad_fc, pred_grad_rho_fc,
                         true_grad_grho_fc, pred_grad_grho_fc,
                         model_name, title)


def plot_heatmap_fv_g_gg(x, y, true_fc, pred_fc, true_grad, pred_grad,
                         true_grad_grho, pred_grad_grho, model_name,
                         title=None):

    xv, yv = jnp.meshgrid(x, y, indexing='ij')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))

    good_title = False
    if len(model_name) < 3:
        good_title = True

    if title is not None:
        fig.suptitle(f'{title}')
    error_val = np.abs(true_fc - pred_fc)
    im1 = ax1.pcolormesh(xv, yv, error_val, shading='auto')
    ax1.set_title(f'Abs. Error for {model_name}')
    ax1.set_xlabel('s')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_yscale('log')
    cbar1 = fig.colorbar(im1, ax=ax1)
    if error_val.min() < 1e-4:
        cbar1.ax.yaxis.get_offset_text().set_x(4.5)

    error_grad = np.abs(true_grad - pred_grad)
    im2 = ax2.pcolormesh(xv, yv, error_grad)
    if not good_title:
        ax2.set_title(f'Abs. Error for {model_name} grad_rho')
    else:
        ax2.set_title(fr'Abs. Error for $\delta${model_name}/$\delta\rho$')
    ax2.set_xlabel('s')
    ax2.set_ylabel(r'$\rho$')
    ax2.set_yscale('log')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.ax.yaxis.set_offset_position('right')

    error_grad_grho = np.abs(true_grad_grho - pred_grad_grho)
    im3 = ax3.pcolormesh(xv, yv, error_grad_grho)
    if not good_title:
        ax3.set_title(f'Abs. Error for {model_name} grad_grho')
    else:
        tit = fr'''Abs. Error for $\delta$ {model_name}/$\delta(\nabla\rho)$'''
        ax3.set_title(tit)
    ax3.set_xlabel('s')
    ax3.set_ylabel(r'$\rho$')
    ax3.set_yscale('log')
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.ax.yaxis.set_offset_position('right')

    plt.tight_layout()
    plt.show()
