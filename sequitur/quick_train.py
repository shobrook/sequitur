# Standard Library
from statistics import mean

# Third Party
import torch
from torch.nn import MSELoss


###########
# UTILITIES
###########


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    if model.__name__ in ("LINEAR_AE", "LSTM_AE"):
        return model(train_set[-1].shape[-1], encoding_dim, **kwargs)
    elif model.__name__ == "CONV_LSTM_AE":
        if len(train_set[-1].shape) == 3:  # 2D elements
            return model(train_set[-1].shape[-2:], encoding_dim, **kwargs)
        elif len(train_set[-1].shape) == 4:  # 3D elements
            return model(train_set[-1].shape[-3:], encoding_dim, **kwargs)


def train_model(
    model, train_set, verbose, lr, epochs, denoise, clip_value, device=None
):
    if device is None:
        device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(reduction="sum")

    mean_losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []
        for x in train_set:
            x = x.to(device)
            optimizer.zero_grad()

            # Forward pass
            x_prime = model(x)

            loss = criterion(x_prime, x)

            # Backward pass
            loss.backward()

            # Gradient clipping on norm
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")

    return mean_losses


def get_encodings(model, train_set, device=None):
    if device is None:
        device = get_device()
    model.eval()
    encodings = [model.encoder(x.to(device)) for x in train_set]
    return encodings


######
# MAIN
######


def quick_train(
    model,
    train_set,
    encoding_dim,
    verbose=False,
    lr=1e-3,
    epochs=50,
    clip_value=1,
    denoise=False,
    device=None,
    **kwargs,
):
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    losses = train_model(
        model, train_set, verbose, lr, epochs, denoise, clip_value, device
    )
    encodings = get_encodings(model, train_set, device)

    return model.encoder, model.decoder, encodings, losses
