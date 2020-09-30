# Standard Library
from statistics import mean

# Third Party
import torch
from torch.nn import MSELoss

# Local Modules
from models import StackedAE, RecurrentAE, RecurrentConvAE


###########
# UTILITIES
###########


def instantiate_model(model, train_set, encoding_dim, **kwargs):
    if isinstance(model, (StackedAE, RecurrentAE)):
        return model(train_set.shape[-1], encoding_dim, **kwargs)
    elif isinstance(model, RecurrentConvAE):
        # TODO: Handle in_channels != 1
        if len(train_set.shape) == 4: # 2D elements
            return model(train_set.shape[-2:], encoding_dim, )
        elif len(train_set.shape) == 5: # 3D elements
            return

def train_model(model, train_set, verbose, lr, epochs):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MSELoss(size_average=False)

    mean_losses = []
    for epoch in range(1, epochs + 1):
        model.train()

        # # Reduces learning rate every 50 epochs
        # if not epoch % 50:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = lr * (0.993 ** epoch)

        losses = []
        for i in range(train_set.shape[0]):
            optimizer.zero_grad()

            # Forward pass
            # x = train_set[i].to(device)
            x = train_set[i]
            x_prime = model(x)

            loss = criterion(x_prime, x)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_loss = mean(losses)
        mean_losses.append(mean_loss)

        if verbose:
            print(f"Epoch: {epoch}, Loss: {mean_loss}")

    return mean_losses


def get_encodings(model, train_set):
    model.eval()

    encodings = []
    for i in range(train_set.shape[0]):
        x = train_set[i]
        encodings.append(model.encoder(x))

    return encodings


######
# MAIN
######


def quick_train(model, train_set, encoding_dim, verbose=False, lr=1e-3,
                epochs=50, **kwargs):
    model = instantiate_model(model, train_set, encoding_dim, **kwargs)
    losses = train_model(model, train_set, verbose, lr, epochs)
    encodings = get_encodings(model, train_set)

    return model.encoder, model.decoder, encodings, losses
