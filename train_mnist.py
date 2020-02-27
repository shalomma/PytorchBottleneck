import torch
import numpy as np
import matplotlib.pyplot as plt
import simplebinmi


class TrainConfig:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer


class Train:
    def __init__(self, config):
        self.config = config
        self.epochs = 5000
        self.mi_cycle = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.losses = dict()
        self.running_mis_xt = dict()
        self.running_mis_ty = dict()
        for phase in ['train', 'test']:
            self.losses[phase] = []
            self.running_mis_xt[phase] = []
            self.running_mis_ty[phase] = []

        self.n_classes = 10

    def get_class_masks(self, loader):
        samples_split = dict()
        for phase in ['train', 'test']:
            samples_split[phase] = {}
            classes = loader[phase].dataset.targets.cpu().detach().numpy()
            for i in range(self.n_classes):
                samples_split[phase][i] = classes == i
        return samples_split

    def run(self, loader):
        class_masks = self.get_class_masks(loader)

        bin_size = 0.5
        nats2bits = 1.0 / np.log(2)

        for i in range(self.epochs):
            to_print = ''
            for phase in ['train', 'test']:
                phase_loss = 0.0
                phase_labels = torch.tensor([], dtype=torch.long).to(self.device)
                phase_outputs = torch.tensor([]).to(self.device)

                for inputs, labels in loader[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if phase == 'train':
                        self.config.model.train()
                    else:
                        self.config.model.eval()

                    self.config.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, hiddens = self.config.model(inputs)
                        loss = self.config.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.config.optimizer.step()

                    phase_loss += loss.item()
                    phase_labels = torch.cat((phase_labels, labels))
                    phase_outputs = torch.cat((phase_outputs, outputs))

                if i % self.mi_cycle == 0:
                    running_mi_xt = []
                    running_mi_ty = []
                    _, hiddens = self.config.model(loader[phase].dataset.data)
                    for j in range(len(self.config.model.hidden_sizes)):
                        activity = hiddens[j].detach().numpy()
                        binxm, binym = simplebinmi.bin_calc_information2(class_masks[phase], activity, bin_size)
                        running_mi_xt.append(nats2bits * binxm)
                        running_mi_ty.append(nats2bits * binym)

                    self.running_mis_xt[phase].append(running_mi_xt)
                    self.running_mis_ty[phase].append(running_mi_ty)

                n = float(len(loader[phase].dataset))
                loss = phase_loss / n
                acc = (phase_labels == phase_outputs.argmax(dim=1)).sum() / n
                to_print += f'{phase}: loss {loss:>.4f} - acc {acc:>.4f} \t'
                self.losses[phase].append(loss)
            print(f'Epoch {i:>4}: {to_print}')

    def plot_losses(self):
        plt.figure()
        for phase in ['train', 'test']:
            plt.plot(self.losses[phase], label=phase)
        plt.legend()
        plt.show()
        plt.savefig('losses.png')

    def plot_info_plan(self, phase):
        plt.figure()
        plt.title(phase)
        plt.plot(self.running_mis_xt[phase])
        plt.plot(self.running_mis_ty[phase])
        plt.show()
        plt.savefig(f'plot_{phase}.png')

        running_mis_xt = np.array(self.running_mis_xt[phase])
        running_mis_ty = np.array(self.running_mis_ty[phase])
        plt.figure()
        plt.title(f'IP - {phase}')
        c = list(range(len(running_mis_xt[:, 0])))
        for j in range(len(self.config.model.hidden_sizes)):
            plt.scatter(running_mis_xt[:, j], running_mis_ty[:, j], c=c, cmap='plasma', s=20, alpha=0.85, zorder=1)
        for j in range(len(running_mis_xt[:, 0])):
            plt.plot(running_mis_xt[j, :], running_mis_ty[j, :], alpha=0.1, zorder=0)
        plt.colorbar()
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.show()
        plt.savefig(f'IP_{phase}.png')
