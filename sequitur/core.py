# Standard Library
from statistics import mean

# Third Party
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

# Local Modules
from .autoencoders import RAE, SAE


###########
# UTILITIES
###########


def prepare_dataset(sequences):
    if type(sequences) == list:
        dataset = []
        for sequence in sequences:
            updated_seq = []
            for vec in sequence:
                if type(vec) == list:
                    updated_seq.append([float(elem) for elem in vec])
                else: # Sequence is 1-D
                    updated_seq.append([float(vec)])

            dataset.append(torch.tensor(updated_seq))
    elif type(sequences) == torch.tensor:
        dataset = [sequences[i] for i in range(len(sequences))]

    shape = torch.stack(dataset).shape
    assert(len(shape) == 3)

    return dataset, shape[1], shape[2]


def train_model(model, dataset, lr, epochs, logging):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    # criterion = CrossEntropyLoss()
    criterion = MSELoss(size_average=False)

    for epoch in range(1, epochs + 1):
        model.train()

        # Reduces learning rate every 50 epochs
        if not epoch % 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (0.993 ** epoch)

        losses, embeddings = [], []
        for seq_true in dataset:
            optimizer.zero_grad()

            # Forward pass
            seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

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


def QuickEncode(sequences, embedding_dim, logging=False, lr=1e-3, epochs=500):
    dataset, seq_len, num_features = prepare_dataset(sequences)
    model = RAE(seq_len, num_features, embedding_dim)
    embeddings, f_loss = train_model(model, dataset, lr, epochs, logging)

    return model.encoder, model.decoder, embeddings, f_loss


if __name__ == "__main__":
    sequences = [[1, 4, 12, 13], [9, 6, 2, 1], [3, 3, 14, 11]]
    encoder, decoder, embeddings, f_loss = QuickEncode(
        sequences,
        embedding_dim=2,
        logging=True
    )

    test_encoding = encoder(torch.tensor([[4.0], [5.0], [6.0], [7.0]]))
    test_decoding = decoder(test_encoding)

    print()
    print(test_encoding)
    print(test_decoding)
