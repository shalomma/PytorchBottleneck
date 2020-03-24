import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Plotter:
    def __init__(self, trainer):
        self.trainer = trainer
        self.n_layers = self.trainer.config.model.n_layers

    def plot_losses(self):
        plt.figure()
        for phase in ['train', 'test']:
            plt.plot(self.trainer.losses[phase], label=phase)
        plt.legend()
        plt.savefig('losses.png')
        plt.show()

    def plot_accuracy(self):
        plt.figure()
        for phase in ['train', 'test']:
            plt.plot(self.trainer.accuracy[phase], label=phase)
        plt.legend()
        plt.savefig('acc.png')
        plt.show()

    def format_epochs(self, x, pos):
        return int(x * self.trainer.mi_cycle)

    def plot_info_plan(self, phase):
        running_mis_xt = np.array(self.trainer.running_mis_xt[phase])
        running_mis_ty = np.array(self.trainer.running_mis_ty[phase])

        fig, ax = plt.subplots()
        plt.title(phase)
        for i in range(self.n_layers):
            plt.plot(running_mis_xt[:, i], label=f'{i}')
        plt.legend()
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_epochs))
        plt.ylabel('I(X;T)')
        plt.savefig(f'plot_{phase}.png')
        plt.show()

        plt.figure()
        plt.title(f'IP - {phase}')
        c = list(range(len(running_mis_xt[:, 0])))
        for j in range(self.n_layers):
            plt.scatter(running_mis_xt[:, j], running_mis_ty[:, j], c=c, cmap='plasma', s=20, alpha=0.85, zorder=1)
        for j in range(len(running_mis_xt[:, 0])):
            plt.plot(running_mis_xt[j, :], running_mis_ty[j, :], alpha=0.1, zorder=0)

        cbar = plt.colorbar(format=ticker.FuncFormatter(self.format_epochs))
        cbar.set_label('Epochs')

        plt.xlabel('I(X;T)')
        plt.ylabel('I(T;Y)')
        plt.savefig(f'IP_{phase}.png')
        plt.show()
