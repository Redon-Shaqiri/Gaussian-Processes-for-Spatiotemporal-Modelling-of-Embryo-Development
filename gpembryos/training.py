import gpytorch as gp
import torch
import copy
from gpembryos.logging_utils import info

import jax
import optax as ox
import gpjax as gpx

def exact_gp_training(model,
                      likelihood,
                      x_train_standardised,
                      y_train_standardised,
                      x_validation,
                      y_validation,
                      training_iterations = 100,
                      learning_rate = 0.1,
                      early_stoppage_wait = 20,
                      has_scheduler = False):
    model.train()
    likelihood.train()

    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

    if has_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.975)



    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_losses = []
    validation_nlpds = []
    validation_epochs = []
    learning_rates = []

    best_validation_nlpd = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_model = None
    best_likelihood = None

    for i in range(training_iterations):
        model.train()
        likelihood.train()
        optimiser.zero_grad()
        output = model(x_train_standardised)
        loss = -mll(output, y_train_standardised)
        loss.backward()
        optimiser.step()
        training_losses.append(loss.item())

        if has_scheduler:
            scheduler.step()
            learning_rates.append(optimiser.param_groups[0]["lr"])

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gp.settings.fast_pred_var(), gp.settings.prior_mode(False):
            validation_output = likelihood(model(x_validation))
            validation_nlpd = (-validation_output.log_prob(y_validation)/y_validation.size(0))
            validation_nlpds.append(validation_nlpd.item())
            validation_epochs.append(i)

        if validation_nlpd.item() < best_validation_nlpd:
            best_validation_nlpd = validation_nlpd.item()
            best_epoch = i
            epochs_without_improvement = 0
            best_model = copy.deepcopy(model.state_dict())
            best_likelihood = copy.deepcopy(likelihood.state_dict())
        else:
            epochs_without_improvement = epochs_without_improvement + 1

        info(f"Epoch {i}: training loss={loss.item():.6f}, "f"validation NLPD={validation_nlpd.item():.6f}")

        if epochs_without_improvement >= early_stoppage_wait:
            info(f"Early stoppage at epoch {i}")
            break

    if best_model is not None:
        model.load_state_dict(best_model)
        likelihood.load_state_dict(best_likelihood)

    info(f"Best model from epoch {best_epoch}")
    info(f"Best validation NLPD {best_validation_nlpd:.6f}")

    results = {
        "model": model,
        "likelihood": likelihood,
        "training_losses": training_losses,
        "validation_nlpds": validation_nlpds,
        "validation_epochs": validation_epochs,
        "best_epoch": best_epoch,
        "best_validation_nlpd": best_validation_nlpd,
        }
    if has_scheduler:
        results["learning_rates"] = learning_rates

    return results





def variational_gp_training(
    model,
    likelihood,
    x_train_standardised,
    y_train_standardised,
    x_validation,
    y_validation,
    training_iterations=100,
    learning_rate=0.1,
    early_stoppage_wait=200,
    has_scheduler=False,
    batch_size=None
):
    model.train()
    likelihood.train()


    if batch_size is None:
        batch_size = x_train_standardised.size(0)

    dataset = torch.utils.data.TensorDataset(x_train_standardised, y_train_standardised)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if has_scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.975)


    mll = gp.mlls.VariationalELBO(likelihood, model, num_data=x_train_standardised.size(0))

    training_losses = []
    validation_nlpds = []
    validation_epochs = []
    learning_rates = []


    best_validation_nlpd = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_model = None
    best_likelihood = None


    for i in range(training_iterations):

        model.train()
        likelihood.train()

        epoch_loss = 0.0

        for x_batch, y_batch in loader:

            optimiser.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        training_losses.append(epoch_loss)

        if has_scheduler:

            scheduler.step()
            learning_rates.append(optimiser.param_groups[0]["lr"])


        model.eval()
        likelihood.eval()

        with torch.no_grad():

            validation_output = likelihood(model(x_validation))
            validation_nlpd = (-validation_output.log_prob(y_validation)/y_validation.size(0))
            validation_nlpd_value = validation_nlpd.item()
            validation_nlpds.append(validation_nlpd_value)
            validation_epochs.append(i)

        if validation_nlpd_value < best_validation_nlpd:
            best_validation_nlpd = validation_nlpd_value
            best_epoch = i
            epochs_without_improvement = 0
            best_model = copy.deepcopy(model.state_dict())
            best_likelihood = copy.deepcopy(likelihood.state_dict())

        else:
            epochs_without_improvement += 1


        if has_scheduler:
            current_lr = optimiser.param_groups[0]["lr"]
            info(
                f"Epoch {i}: "
                f"training loss={epoch_loss:.6f}, "
                f"validation NLPD={validation_nlpd_value:.6f}, "
                f"learning rate={current_lr:.8f}"
            )

        else:
            info(
                f"Epoch {i}: "
                f"training loss={epoch_loss:.6f}, "
                f"validation NLPD={validation_nlpd_value:.6f}"
            )


        if epochs_without_improvement >= early_stoppage_wait:

            info(
                f"Early stoppage at epoch {i}"
            )

            break

  

    if best_model is not None:

        model.load_state_dict(best_model)

        likelihood.load_state_dict(best_likelihood)

    info(
        f"Best model from epoch {best_epoch}"
    )

    info(
        f"Best validation NLPD "
        f"{best_validation_nlpd:.6f}"
    )

    results = {
        "model": model,
        "likelihood": likelihood,
        "training_losses": training_losses,
        "validation_nlpds": validation_nlpds,
        "validation_epochs": validation_epochs,
        "best_epoch": best_epoch,
        "best_validation_nlpd": best_validation_nlpd,
    }

    if has_scheduler:
        results["learning_rates"] = learning_rates

    return results


