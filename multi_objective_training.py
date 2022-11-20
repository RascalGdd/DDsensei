import torch
from models.min_norm_solvers import MinNormSolver, gradient_normalizers
import config

opt = config.read_arguments(train=True)


def multi_objective(label, optimizer, model, image2, image):
    loss_data = {}
    grads = {}
    scale = {}
    tasks = ["netDu", "netD", "lpips"]
    loss_netDu = 0
    loss_netD = 0
    loss_lpips = 0


    optimizer.zero_grad()
    # First compute representations (z)

    generated = model(image, label, "generate", losses_computer=None)
    rep = generated
    print("type of rep is list or not?", type(rep))
    rep_grad = rep.clone()
    rep_grad.requires_grad = True



# for netDu
    optimizer.zero_grad()
    model.module.netG.zero_grad()
    model.module.netD.zero_grad()
    model.module.netDu.zero_grad()
    loss_netDu = model(rep_grad, label, "losses_G_ori2", image2=image2, losses_computer=None).mean()
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
    loss_netD, _ = model(rep_grad, label, "losses_G_multi", image2=image2, losses_computer=None)
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
    _, loss_lpips = model(rep_grad, label, "losses_G_multi", image2=image2, losses_computer=None)
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
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
    for i, t in enumerate(tasks):
        scale[t] = float(sol[i])

    optimizer.zero_grad()
    loss_netDu = model(rep_grad, label, "losses_G_ori2", image2=image2, losses_computer=None).mean()
    loss_netDu = loss_netDu * scale["netDu"]
    loss_netD, loss_lpips = model(rep_grad, label, "losses_G_multi", image2=image2, losses_computer=None)
    loss_netD = loss_netD * scale["netD"]
    loss_lpips = loss_lpips * scale["lpips"]

    print("loss_netDu", loss_netDu)
    print("loss_netD", loss_netD)
    print("loss_lpips", loss_lpips)

    loss = loss_netDu + loss_netD + loss_lpips
    loss.backward()
    optimizer.step()





