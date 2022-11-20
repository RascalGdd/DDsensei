import torch
from models.min_norm_solvers import MinNormSolver, gradient_normalizers
import config

opt = config.read_arguments(train=True)


def multi_objective(label, optimizer, model, image2, losses_computer):
    loss_data = {}
    grads = {}
    scale = {}
    tasks = ["netDu", "netD", "lpips"]

    optimizer.zero_grad()
    # First compute representations (z)

    generated = model(None, label, "generate_fortraining", None, None)
    rep = generated
    rep_grad = rep.clone()
    rep_grad.requires_grad = True

# for netDu
    optimizer.zero_grad()
    model.module.netG.zero_grad()
    model.module.netD.zero_grad()
    model.module.netDu.zero_grad()
    loss_netDu = model(rep_grad, None, "losses_multi_netDu", losses_computer, None).mean()
    loss_data["netDu"] = loss_netDu
    loss_netDu.backward()
    grads["netDu"] = []
    grads["netDu"].append(rep_grad.grad.clone())
    rep_grad.grad.zero_()

# for netD
    optimizer.zero_grad()
    model.module.netG.zero_grad()
    model.module.netD.zero_grad()
    model.module.netDu.zero_grad()
    loss_netD = model(rep_grad, None, "losses_multi_netD", losses_computer, None)
    loss_data["netD"] = loss_netD
    loss_netD.backward()
    grads["netD"] = []
    grads["netD"].append(rep_grad.grad.clone())
    rep_grad.grad.zero_()

# for lpips
    optimizer.zero_grad()
    model.module.netG.zero_grad()
    model.module.netD.zero_grad()
    model.module.netDu.zero_grad()
    loss_lpips = model(rep_grad, None, "losses_multi_lpips", losses_computer, image2)
    loss_data["lpips"] = loss_lpips
    loss_lpips.backward()
    grads["lpips"] = []
    grads["lpips"].append(rep_grad.grad.clone())
    rep_grad.grad.zero_()

    gn = gradient_normalizers(grads, loss_data, "loss+")
    for t in tasks:
        for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]

    # Frank-Wolfe iteration to compute scales.
    # print(len(grads))
    # print(grads["netDu"][0].shape)
    # print(grads["netD"][0].shape)
    # print(grads["lpips"][0].shape)
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
    for i, t in enumerate(tasks):
        scale[t] = float(sol[i])
    print(scale)
    optimizer.zero_grad()
    loss_netDu = model(None, label, "losses_G_ori2", losses_computer, image2).mean()
    loss_netDu = loss_netDu * scale["netDu"]
    loss_netD, loss_lpips = model(None, label, "losses_G_multi", losses_computer, image2)
    loss_netD = loss_netD * scale["netD"]
    loss_lpips = loss_lpips * scale["lpips"]

    print("loss_netDu", loss_netDu)
    print("loss_netD", loss_netD)
    print("loss_lpips", loss_lpips)

    loss = loss_netDu + loss_netD + loss_lpips
    loss.backward()
    optimizer.step()






