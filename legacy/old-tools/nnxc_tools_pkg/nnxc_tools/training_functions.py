# FUNCTIONS TO TRAIN THE MODEL
import jax.numpy as jnp
import jax
import equinox as eqx
from functools import partial
from modules.reference_functionals import pw92c_unpolarized_scalar, lda_x
import warnings


# Modified training loop
@eqx.filter_jit
def make_step_vg(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_values_and_grad(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_values_and_grad(model, inputs, ref, ref_grad):
    """
    Loss with the value and the gradient for the loss, for Fc
    """
    # Compute predicted values
    pred = jax.vmap(model)(inputs)

    # Compute predicted gradients w.r.t. rho and grad_rho
    pred_grad_rho, pred_grad_grad_rho = \
        jax.vmap(partial(compute_derivatives, model))(inputs)

    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_grad_rho = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_grad_rho - ref_grad_grad_rho) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_g(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_grad(model, inputs, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_grad(model, inputs, ref_grad):
    """
    Loss with the loss in the gradient, for Fc
    """

    # Compute predicted gradients w.r.t. rho and grad_rho
    pred_grad_rho, pred_grad_grad_rho = \
        jax.vmap(partial(compute_derivatives, model))(inputs)

    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_grad_rho = ref_grad

    # Debugging: print the structure of ref_grad and pred_grad
    print(f"ref_grad_rho shape: {ref_grad_rho.shape}, ref_grad_grad_rho shape: {ref_grad_grad_rho.shape}")
    print(f"pred_grad_rho shape: {pred_grad_rho.shape}, pred_grad_grad_rho shape: {pred_grad_grad_rho.shape}")

    # Compute MSE for function values and each derivative
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_grad_rho - ref_grad_grad_rho) ** 2)

    print(loss_grad_rho, loss_grad_grad_rho)

    # Combine losses
    total_loss = loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_v(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_values(model, inputs, ref)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_values(model, inputs, ref):
    """
    Loss with the loss in the gradient, for Fc
    """
    # Compute predicted values
    pred = jax.vmap(model)(inputs)
    # Compute MSE for function values and each derivative
    loss_vals = jnp.mean((pred - ref) ** 2)

    return loss_vals


@eqx.filter_jit
def make_step_erho_fx_dsigma(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_vg_erho_fx_dsigma(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_vg_erho_fx_dsigma(model, inputs, ref, ref_grad):
    # Compute predicted values
    pred = jax.vmap(model)(inputs)
    inputs = jnp.atleast_1d(inputs)  # Ensure inputs is at least 1D

    if inputs.ndim == 1 and inputs.shape[0] == 2:
        # Case when inputs is (2,)
        rho, grho = inputs
        sigma = grho**2
        sigma_inputs = jnp.array([rho, sigma])  # Keep it (2,)
    else:
        # Case when inputs is (N,2)
        rho = inputs[:, 0]
        grho = inputs[:, 1]
        sigma = grho**2
        sigma_inputs = jnp.stack([rho, sigma], axis=-1)  # Keep it (N,2)

    # Compute predicted gradients of rho*e_ueg(rho)*model w.r.t. rho and sigma
    def e_rho_func(sigma_inputs):
        """ This has sigma a input, to derivate againts sigma"""
        rho, sigma = sigma_inputs
        grho = jnp.sqrt(sigma)
        inputs = jnp.stack([rho, grho], axis=-1)
        return lda_x(rho) * rho * model(inputs)
    pred_grad_rho, pred_grad_sigma = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(sigma_inputs)
    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_sigma = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_sigma - ref_grad_sigma) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_erho_fc_dsigma(model, inputs, ref, ref_grad, opt_state,
                             optimizer):
    loss, grad = __loss_vg_erho_fc_dsigma(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_vg_erho_fc_dsigma(model, inputs, ref, ref_grad):
    # Compute predicted values
    pred = jax.vmap(model)(inputs)

    inputs = jnp.atleast_1d(inputs)  # Ensure inputs is at least 1D

    if inputs.ndim == 1 and inputs.shape[0] == 2:
        # Case when inputs is (2,)
        rho, grho = inputs
        sigma = grho**2
        sigma_inputs = jnp.array([rho, sigma])  # Keep it (2,)
    else:
        # Case when inputs is (N,2)
        rho = inputs[:, 0]
        grho = inputs[:, 1]
        sigma = grho**2
        sigma_inputs = jnp.stack([rho, sigma], axis=-1)  # Keep it (N,2)

    # Compute predicted gradients of rho*e_ueg(rho)*model w.r.t. rho and sigma
    def e_rho_func(sigma_inputs):
        rho, sigma = sigma_inputs
        grho = jnp.sqrt(sigma)
        inputs = jnp.stack([rho, grho], axis=-1)
        return pw92c_unpolarized_scalar(rho) * rho * model(inputs)
    pred_grad_rho, pred_grad_sigma = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(sigma_inputs)

    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_sigma = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_sigma - ref_grad_sigma) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_erho_fx_dgrho(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_vg_erho_fx_dgrho(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_vg_erho_fx_dgrho(model, inputs, ref, ref_grad):
    """ Model takes rho and grad_rho as inputs. Ref_grad is computed with
    respect to rho and grad_rho."""
    # Compute predicted values
    pred = jax.vmap(model)(inputs)
    inputs = jnp.atleast_1d(inputs)  # Ensure inputs is at least 1D

    # Compute predicted gradients of rho*e_ueg(rho)*model
    #  w.r.t. rho and grad_rho
    def e_rho_func(grho_inputs):
        """This has rho and grad_rho as inputs, to derivate againts grad_rho"""
        rho, _ = grho_inputs
        return lda_x(rho) * rho * model(grho_inputs)
    pred_grad_rho, pred_grad_grho = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(inputs)
    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_grho = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_grho - ref_grad_grho) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_erho_fc_dgrho(model, inputs, ref, ref_grad, opt_state,
                            optimizer):
    loss, grad = __loss_vg_erho_fc_dgrho(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_vg_erho_fc_dgrho(model, inputs, ref, ref_grad):
    # Compute predicted values
    pred = jax.vmap(model)(inputs)

    # Compute predicted gradients of rho*e_ueg(rho)*model wrt rho and grad_rho
    def e_rho_func(grho_inputs):
        rho, _ = grho_inputs
        return pw92c_unpolarized_scalar(rho) * rho * model(grho_inputs)
    pred_grad_rho, pred_grad_grho = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(inputs)

    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_grho = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_grho - ref_grad_grho) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_erho_fxSIG(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_vg_erho_fxSIG(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def __loss_vg_erho_fxSIG(model, inputs, ref, ref_grad):
    """ Model takes rho and sigma as inputs."""
    # Compute predicted values
    pred = jax.vmap(model)(inputs)
    inputs = jnp.atleast_1d(inputs)  # Ensure inputs is at least 1D

    # Compute predicted gradients of rho*e_ueg(rho)*model w.r.t. rho and sigma
    def e_rho_func(sigma_inputs):
        """ This has sigma a input, to derivate againts sigma"""
        rho, sigma = sigma_inputs
        return lda_x(rho) * rho * model(sigma_inputs)
    pred_grad_rho, pred_grad_sigma = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(inputs)
    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_sigma = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_sigma - ref_grad_sigma) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


@eqx.filter_jit
def make_step_erho_fcSIG(model, inputs, ref, ref_grad, opt_state, optimizer):
    loss, grad = __loss_vg_erho_fcSIG(model, inputs, ref, ref_grad)
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_value_and_grad
def __loss_vg_erho_fcSIG(model, inputs, ref, ref_grad):
    """ Model takes rho and sigma as inputs."""
    # Compute predicted values
    pred = jax.vmap(model)(inputs)
    inputs = jnp.atleast_1d(inputs)  # Ensure inputs is at least 1D

    # Compute predicted gradients of rho*e_ueg(rho)*model w.r.t. rho and sigma
    def e_rho_func(sigma_inputs):
        """ This has sigma a input, to derivate againts sigma"""
        rho, sigma = sigma_inputs
        return pw92c_unpolarized_scalar(rho) * rho * model(sigma_inputs)
    pred_grad_rho, pred_grad_sigma = \
        jax.vmap(partial(compute_derivatives, e_rho_func))(inputs)
    # ref_grad is a tuple, so we need to unpack it
    ref_grad_rho, ref_grad_sigma = ref_grad

    # Compute MSE for function values and each derivative
    loss_values = jnp.mean((pred - ref) ** 2)
    loss_grad_rho = jnp.mean((pred_grad_rho - ref_grad_rho) ** 2)
    loss_grad_grad_rho = jnp.mean((pred_grad_sigma - ref_grad_sigma) ** 2)

    # Combine losses
    total_loss = loss_values + loss_grad_rho + loss_grad_grad_rho

    return total_loss


def loop(model, model_name, inputs, optimizer, ref=None, ref_grad=None,
         epochs=5000, printevery=2500, erho=False, savemodels=False,
         deriv_wrt=None):

    print(f"Starting training for model: {model_name}")

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    if ref is None and ref_grad is None:
        raise ValueError('Please, insert some reference values in loop')
    if erho:
        model_name_lower = model.name.lower()
        if 'sig' in model_name_lower:
            if 'fx' in model_name_lower:
                make_step = make_step_erho_fxSIG
            elif 'fc' in model_name_lower:
                make_step = make_step_erho_fcSIG
        elif deriv_wrt is None:
            warnings.warn('We are assuming reference derivatives are wrt rho\
 and sigma (i.e. obtained with gtv_FX/C_grid_vgrho_dvsig)')
            if 'fx' in model_name_lower:
                make_step = make_step_erho_fxSIG
            elif 'fc' in model_name_lower:
                make_step = make_step_erho_fcSIG
        elif deriv_wrt == 'sigma':
            if 'fx' in model_name_lower:
                make_step = make_step_erho_fx_dsigma
            elif 'fc' in model_name_lower:
                make_step = make_step_erho_fc_dsigma
            else:
                raise ValueError('Model name not recognized for erho training')
        elif deriv_wrt == 'grad_rho':
            if 'fx' in model_name_lower:
                make_step = make_step_erho_fx_dgrho
            elif 'fc' in model_name_lower:
                make_step = make_step_erho_fc_dgrho
            else:
                raise ValueError('Model name not recognized for erho training')
        else:
            raise NotImplementedError(
                f'Training with ferivative with respect to {deriv_wrt}  and \
erho not implemented')
    elif ref_grad is None:
        make_step = make_step_v
    elif ref is None:
        make_step = make_step_g
    else:
        make_step = make_step_vg

    models = []
    losses_steps = []
    steps = []

    for step in range(epochs):
        loss, model, opt_state = make_step(model, inputs, ref, ref_grad,
                                           opt_state, optimizer)
        lossi = loss.item()
        losses.append(lossi)
        if step % printevery == 0:
            print(f'Epoch {step}: Loss = {lossi}')
            if savemodels:
                models.append(model)
                losses_steps.append(lossi)
                steps.append(step)

    if not savemodels:
        return model, losses
    else:
        return models, losses, losses_steps, steps


# UTILS

@eqx.filter_jit
def compute_derivatives(model, inputs):
    """Compute gradient w.r.t. both components (rho and grad_rho)

    rho = inputs[0]
    grad_rho = inputs[1]
    """

    # x is the value wrt we derivate.
    grad_rho = jax.grad(lambda x:
                        model(jnp.array([x, inputs[1]])))(inputs[0])
    grad_grad_rho = jax.grad(lambda x:
                             model(jnp.array([inputs[0], x])))(inputs[1])
    return (grad_rho, grad_grad_rho)
