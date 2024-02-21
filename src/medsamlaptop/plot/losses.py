import matplotlib.pyplot as plt

def plot_and_save_loss(train_losses, saving_dir):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(train_losses)
    ax.set_title("Dice + Binary Cross Entropy + IoU Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(saving_dir / "train_loss.png", dpi=300)
    plt.close()