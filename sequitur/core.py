# Standard Library
from statistics import mean

# Third Party
from torch import tensor, optim as optimizers, device, cuda
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy

# Local Modules
from autoencoders import RAE, SAE


###########
# UTILITIES
###########


# TODO
def prepare_dataset(sequences):
    if type(sequences) == list:
        pass
    elif type(sequences) == tensor:
        pass

    # Create a Variable() for each sequence

    return dataset, seq_len, num_features


def train_model(model, dataset, lr, epochs, logging):
    device = device("cuda" if cuda.is_available() else "cpu")
    optimizer = optimizers.Adam(model.parameters())
    criterion = binary_cross_entropy

    for epoch in range(1, epochs + 1):
        model.train()

        # Reduces learning rate every 50 epochs
        if not epoch % 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (0.993 ** epoch)

        losses, embeddings = [], []
        for seq in dataset:
            optimizer.zero_grad()

            # Forward pass
            sequence.to(device)
            seq_pred = model(sequence)

            loss = criterion(seq_pred, sequence)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            embeddings.append(seq_pred)

        if logging:
            print("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))

    return embeddings, mean(losses)


#########
# EXPORTS
#########


def QuickEncode(sequences, embedding_dim, logging=False, lr=1e-3, epochs=100):
    dataset, seq_len, num_features = prepare_dataset(sequences)
    model = RAE(seq_len, num_features, embedding_dim)
    embeddings, f_loss = train_model(model, dataset, lr, epochs, logging)

    return model.encoder, model.decoder, embeddings, f_loss
