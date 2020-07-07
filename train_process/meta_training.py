import torch
from torch import nn

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(batch, learner, regs, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = torch.zeros(data.size(0), dtype=torch.bool)#.byte()
    adaptation_indices[torch.arange(shots*ways) * 2] = 1
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices].long()
    evaluation_data, evaluation_labels = data[~adaptation_indices], labels[~adaptation_indices].long()

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)

    if len(regs) > 0:
        for reg in regs:
            valid_error += reg(learner)

    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def trainingProcessMeta(args, model, opt, data_generators, regs, num_cls, device):
    loss = nn.CrossEntropyLoss(reduction='mean')
    meta_train_error = 0.0
    meta_train_accuracy = 0.0
    opt.zero_grad()
    for task in range(args.meta_batch_size):
        # Compute meta-training loss
        learner = model.clone()
        batch = data_generators.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               regs,
                                                               loss,
                                                               args.adaptation_steps,
                                                               args.shots,
                                                               num_cls,
                                                               device)
        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        meta_train_accuracy += evaluation_accuracy.item()

    # Average the accumulated gradients and optimize
    for p in model.parameters():
        p.grad.data.mul_(1.0 / args.meta_batch_size)
    opt.step()

    return meta_train_error / args.meta_batch_size, meta_train_accuracy / args.meta_batch_size, 