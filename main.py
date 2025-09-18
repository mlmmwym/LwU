import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import math
import types
import time


def mask_precentage(mask):
    all = 0
    all_b = 0
    for index in range(len(mask)):
        item = mask[index]
        # item = torch.tensor(item)
        a = (item == 0.).float().sum()
        b = (item == 1.).float().sum()
        all += a.item() + b.item()
        all_b += b.item()

        zero_ratio = b.item() / (a.item() + b.item())
        print("layer {} remain: {}".format(index+1, zero_ratio))
    print("all parameter:", all)
    print("all forget:", all_b)
    print("forget rate:", all_b / all)
    return all_b / all

def apply_mask(net, keep_masks):

    prunable_layers = filter(lambda layer: isinstance(layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):

        init_values = torch.randn_like(layer.weight.data)
        init_values = torch.empty_like(layer.weight.data)

        nn.init.orthogonal_(init_values, gain=1.0)
        layer.weight.data[keep_mask == 0.] = init_values[keep_mask == 0.]



def layer_train(net, data, label, criterion, datatype):

    lr = 0.02
    # lr = 5e-4

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.95)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)

    nData = data.shape[0]

    if datatype == 1:
        batchSize  = 10
    elif datatype == 2:
        batchSize = 50

    nBatch = int(nData/batchSize)


    predict_y = net(data)

    loss = criterion(predict_y, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return net

def layer_transferability_estimation(net, inputs, targets, criterion, datatype):
    loss_rate_list = []
    epoch = 100
    # epoch = 50
    # epoch = 25
    predict_y = net(inputs)
    loss = criterion(predict_y, targets)
    loss = loss.item()
    layers = get_layers(net)
    for i in range(len(layers)):
        extend_net = copy.deepcopy(net)
        extend_layers = get_layers(extend_net)
        for j in range(len(extend_layers)):
            if j != i:
                extend_layers[j].weight.requires_grad = False
        for k in range(epoch):
            extend_net = layer_train(extend_net, inputs, targets, criterion, datatype)
        _predict_y = extend_net(inputs)
        _loss = criterion(_predict_y, targets)
        _loss = _loss.item()
        loss_rate = (loss - _loss) / loss
        loss_rate_list.append(loss_rate)

    return loss_rate_list

def layer_forget_estimation(layer_forget_rate):

    layer_forget = []
    for index in range(len(layer_forget_rate)):
        forget_rate = (1 - layer_forget_rate[index]) * math.pow(((index + 1) / len(layer_forget_rate)), 1)
        forget_rate = 1 - forget_rate
        layer_forget.append(forget_rate)

    return layer_forget

def get_layers(net):
    layers = []
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            layers.append(layer)
    # layers = layers[:-1]
    return layers

def get_forget_point(keep_ratio_list, delta):
    k_tensor = torch.tensor(keep_ratio_list)
    max_value = torch.max(k_tensor).item()
    # hoeffding_bound = math.sqrt(math.log((1 / delta), math.e) / (2 * len(k_tensor)))
    hoeffding_bound = 0.15
    forget_point = len(k_tensor) - 1
    for index in range(len(k_tensor) - 1):
        a_mean = torch.mean(k_tensor[:index + 1]).item()
        b_mean = torch.mean(k_tensor[index + 1:]).item()
        b = k_tensor[index].item()

        if max_value - b > hoeffding_bound:
        # if a_mean - b_mean > hoeffding_bound:
            forget_point = index
            break

    if forget_point >= len(k_tensor):
        forget_point = len(k_tensor) - 1
    return forget_point
    # return forget_point + 1



def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def SNIP_layers(net, keep_ratio_list, inputs, targets):
    net = copy.deepcopy(net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    index = 0
    keep_masks = []
    for layer in net.modules():
        # print(layer)
        if isinstance(layer, nn.Linear):
            layer_grad = torch.abs(layer.weight_mask.grad)
            layer_scores = torch.cat([torch.flatten(x) for x in layer_grad])
            total_params = len(layer_scores)
            if index < len(keep_ratio_list):
                num_params_to_keep = int(len(layer_scores) * keep_ratio_list[index])
            else:
                num_params_to_keep = len(layer_scores)

            norm_factor = torch.sum(layer_scores)
            layer_scores.div_(norm_factor)

            threshold, _ = torch.topk(layer_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]


            mask = ((layer_grad / norm_factor) >= acceptable_score).float()
            # mask = ((layer_grad / norm_factor) > acceptable_score).float()

            if torch.all(mask == 1) and num_params_to_keep < total_params:

                target_discard = total_params - num_params_to_keep


                eq_mask = torch.isclose(
                    (layer_grad / norm_factor),
                    torch.tensor(acceptable_score, device=layer_grad.device),
                    atol=1e-6
                )


                candidates = torch.nonzero(eq_mask, as_tuple=True)
                flat_indices = torch.arange(eq_mask.numel(), device=eq_mask.device)[eq_mask.view(-1)]


                if len(flat_indices) >= target_discard:
                    selected = flat_indices[torch.randperm(len(flat_indices))[:target_discard]]
                else:

                    selected = flat_indices

                # 将选中的位置置零
                mask_flat = mask.view(-1)
                mask_flat[selected] = 0
                mask = mask_flat.view_as(mask)




            keep_masks.append(mask)
            index += 1
    # forget_point = get_forget_point(keep_ratio_list, 0.9)
    forget_point = get_forget_point(keep_ratio_list, 0.5)
    # forget_point = get_forget_point(keep_ratio_list, 0.1)
    for index in range(len(keep_masks[:forget_point])):
        keep_masks[index] = torch.ones_like(keep_masks[index], dtype=torch.float32)


    return (keep_masks)

def custom_sampling(preq_data, preq_label, fine_samples=200, test_samples=1000):
    fine_samples = int(fine_samples / 2)
    test_samples = int(test_samples / 2)
    n_samples = fine_samples + test_samples
    unique_labels = np.unique(preq_label)

    # sampled_indices = []
    fine_sampled_indices = []
    test_sampled_indices = []

    for label in unique_labels:
        indices = np.where(preq_label == label)[0]

        selected = np.random.choice(indices, size=n_samples, replace=False)
        fine_selected = selected[:fine_samples]
        test_selected = selected[fine_samples:]

        fine_sampled_indices.extend(fine_selected)
        test_sampled_indices.extend(test_selected)

    fine_sampled_indices = np.array(fine_sampled_indices)
    test_sampled_indices = np.array(test_sampled_indices)

    np.random.shuffle(fine_sampled_indices)
    np.random.shuffle(test_sampled_indices)


    return preq_data[fine_sampled_indices], preq_label[fine_sampled_indices], preq_data[test_sampled_indices], preq_label[test_sampled_indices]

def layer_transferability_estimation_mask(net, inputs, targets, criterion, datatype):
    loss_rate_list = []
    epoch = 100
    # epoch = 50
    # epoch = 25
    predict_y = net(inputs)
    loss = criterion(predict_y, targets)
    loss = loss.item()
    layers = get_layers(net)
    delta = 0.5
    hoeffding_bound = math.sqrt(math.log((1 / delta), math.e) / (2 * len(layers)))

    transferability_list = [1] * len(layers)

    extend_net = copy.deepcopy(net)
    extend_layers = get_layers(extend_net)
    for j in range(len(extend_layers)):
        if j != 0:
            extend_layers[j].weight.requires_grad = False
    for k in range(epoch):
        extend_net = layer_train(extend_net, inputs, targets, criterion, datatype)
    _predict_y = extend_net(inputs)
    _loss = criterion(_predict_y, targets)
    _loss = _loss.item()
    loss_rate_max = (loss - _loss) / loss


    for i in reversed(range(len(layers))):
        extend_net = copy.deepcopy(net)
        extend_layers = get_layers(extend_net)
        for j in range(len(extend_layers)):
            if j != i:
                extend_layers[j].weight.requires_grad = False
        for k in range(epoch):
            extend_net = layer_train(extend_net, inputs, targets, criterion, datatype)
        _predict_y = extend_net(inputs)
        _loss = criterion(_predict_y, targets)
        _loss = _loss.item()
        loss_rate = (loss - _loss) / loss
        transferability_list[i] = loss_rate
        if loss_rate_max - loss_rate > hoeffding_bound:
            break


    net = copy.deepcopy(net)
    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            layer.forward = types.MethodType(snip_forward_linear, layer)

    net.zero_grad()
    outputs = net.forward(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()

    index = 0
    keep_masks = []

    for layer in net.modules():
        # print(layer)
        if isinstance(layer, nn.Linear):
            layer_grad = torch.abs(layer.weight_mask.grad)
            layer_scores = torch.cat([torch.flatten(x) for x in layer_grad])
            total_params = len(layer_scores)
            if index < len(transferability_list):
                num_params_to_keep = int(len(layer_scores) * transferability_list[index])
            else:
                num_params_to_keep = len(layer_scores)

            norm_factor = torch.sum(layer_scores)
            layer_scores.div_(norm_factor)

            threshold, _ = torch.topk(layer_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]


            mask = ((layer_grad / norm_factor) >= acceptable_score).float()
            # mask = ((layer_grad / norm_factor) > acceptable_score).float()

            if torch.all(mask == 1) and num_params_to_keep < total_params:

                target_discard = total_params - num_params_to_keep


                eq_mask = torch.isclose(
                    (layer_grad / norm_factor),
                    torch.tensor(acceptable_score, device=layer_grad.device),
                    atol=1e-6
                )


                candidates = torch.nonzero(eq_mask, as_tuple=True)
                flat_indices = torch.arange(eq_mask.numel(), device=eq_mask.device)[eq_mask.view(-1)]


                if len(flat_indices) >= target_discard:
                    selected = flat_indices[torch.randperm(len(flat_indices))[:target_discard]]
                else:

                    selected = flat_indices


                mask_flat = mask.view(-1)
                mask_flat[selected] = 0
                mask = mask_flat.view_as(mask)




            keep_masks.append(mask)
            index += 1

    return keep_masks









def train(preq_data, preq_label, net, datatype):

    nData = preq_data.shape[0]

    if datatype == 1:
        batchSize = 10
    elif datatype == 2:
        batchSize = 50

    nBatch = int(nData / batchSize)
    nEpoch = 100
    generation = 20

    criterion = nn.CrossEntropyLoss()

    lr = 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.95)


    for iEpoch in range(0, nEpoch):

        print(f'Epoch {iEpoch + 1}/{nEpoch}')
        print('-' * 10)

        if iEpoch % generation == 0:
        # if iEpoch == 0:
            print("New generation!")
            print("*" * 10)
            if iEpoch == 0:
                layer_forget_rate = layer_transferability_estimation(net, preq_data, preq_label, criterion, datatype)
                mask = SNIP_layers(net, layer_forget_rate, preq_data, preq_label)


            apply_mask(net, mask)


        for iBatch in range(0, nBatch):
            batchIdx = iBatch + 1
            batchData = preq_data[(batchIdx - 1) * batchSize:batchIdx * batchSize]
            batchLabel = preq_label[(batchIdx - 1) * batchSize:batchIdx * batchSize]

            predict_y = net(batchData)

            loss = criterion(predict_y, batchLabel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('batch: %d/%d, loss = %.4f' % (batchIdx, nBatch, loss.item()))

    return net

def test(test_data, test_label, net):

    predict_y = net(test_data)
    predict_label = torch.argmax(predict_y, dim=1)
    true_label = torch.eq(predict_label, test_label)
    true_label = torch.sum(true_label).item()
    acc = true_label / test_data.shape[0]

    return acc


def main(source_domain, dataset, layers):


    model = "pre_model/" + source_domain + "_MLP" + str(layers) + ".pth"
    net = torch.load(model)

    amazon_datasets = ['books', 'dvd', 'electronics', 'kitchen']
    syn_datasets = ['RBF_concept1', 'Tree_concept1', 'SCD_concept1', 'Sine_concept1', 'SEA_concept1_1', 'SEA_concept1_2', 'SEA_concept1_3']

    amazon_path = "data/AmazonReviews/"
    syn_path = "data/Synthetic/"

    fileName = dataset + '.npz'
    datatype = 0


    if dataset in amazon_datasets:
        datatype = 1
        data = np.load(amazon_path + fileName)
    elif dataset in syn_datasets:
        datatype = 2
        data = np.load(syn_path + fileName)

    preq_data = data["x_train"]
    preq_label = data["y_train"]
    preq_data = torch.FloatTensor(preq_data)
    preq_label = torch.LongTensor(preq_label).view(-1)

    if datatype == 1:
        train_data, train_label, test_data, test_label = custom_sampling(preq_data, preq_label, 200, 1000)
    elif datatype == 2:
        train_data = preq_data[:500]
        train_label = preq_label[:500]
        # train_data = preq_data[:100]
        # train_label = preq_label[:100]
        test_data = preq_data[4001:]
        test_label = preq_label[4001:]

    since = time.time()
    net = train(train_data, train_label, net, datatype)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Training completed! ")

    acc = test(test_data, test_label, net)

    print('Test acc: %.4f' % acc)


main('kitchen', 'books', 5)

