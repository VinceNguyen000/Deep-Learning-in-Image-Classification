import time
import torch
from evaluation import compute_accuracy


# Build a function to train our model
def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, device, scheduler=None,
                scheduler_on='valid_acc'):
    start_time = time.time()
    batch_lost_list, train_accuracy_list, valid_accuracy_list = [], [], []

    # cnt = 0
    for epoch in range(num_epochs):
        #
        # model.train()
        # for batch_idx, (features, targets) in enumerate(train_loader):
        #
        #     features = features.to(device)
        #     targets = targets.to(device)
        #
        #     # ## FORWARD AND BACK PROP
        #     logits, _ = model(features)
        #     loss = torch.nn.functional.cross_entropy(logits, targets)
        #     writer.add_scalar("Loss/train", loss, cnt)
        #     cnt = cnt + 1
        #     optimizer.zero_grad()
        #
        #     loss.backward()
        #
        #     # ## UPDATE MODEL PARAMETERS
        #     optimizer.step()
        #
        #     # ## LOGGING
        #     batch_lost_list.append(loss.item())
        #     if not batch_idx % logging_interval:
        #         print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
        #               f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
        #               f'| Loss: {loss:.4f}')
        # torch.save(model.state_dict(), "./model-vgg-final.ckpt")
        #
        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}%')
            train_accuracy_list.append(train_acc.item())
            valid_accuracy_list.append(valid_acc.item())

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_accuracy_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(batch_lost_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return batch_lost_list, train_accuracy_list, valid_accuracy_list
