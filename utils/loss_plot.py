import matplotlib.pyplot as plt
import os



def plot_losses(opt, losses, scales, name):

    path = os.path.join(opt.checkpoints_dir, opt.name, "multi_losses")
    os.makedirs(path, exist_ok=True)
    fig = plt.figure()
    plt.plot(losses)
    save_path = os.path.join(path, name + "_loss.png")
    fig.savefig(save_path)

    fig2 = plt.figure()
    plt.plot(scales)
    path = os.path.join(opt.checkpoints_dir, opt.name, "multi_losses")
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name + "_scale.png")
    fig2.savefig(save_path)


