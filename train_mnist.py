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

        self.losses = dict()
        self.running_mis_xt = dict()
        self.running_mis_ty = dict()
        for phase in ['train', 'test']:
            self.losses[phase] = []
            self.running_mis_xt[phase] = []
            self.running_mis_ty[phase] = []

        self.n_classes = 10

    def get_label_masks(self, labels):
        classes = labels.detach().numpy()
        samples_split = dict()
        for i in range(self.n_classes):
            samples_split[i] = classes == i
        return samples_split

    def run(self, loader):
        bin_size = 0.07
        nats2bits = 1.0 / np.log(2)

        for i in range(self.epochs):
            to_print = ''
            for phase in ['train', 'test']:
                running_loss = 0.0

                for batch in loader[phase]:
                    inputs, labels = batch[0], batch[1]
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

                    running_loss += loss.item()
                    self.losses[phase].append(loss)
                    # acc = (labels == outputs.argmax(dim=1)).sum() / float(len(inputs))


                    # if i % 10 == 0:
                    #     running_mi_xt = []
                    #     running_mi_ty = []
                    #     for j in range(len(self.config.model.hidden_sizes)):
                    #         activity = hiddens[j].detach().numpy()
                    #         label_masks = self.get_label_masks(labels)
                    #         binxm, binym = simplebinmi.bin_calc_information2(label_masks, activity, bin_size)
                    #         running_mi_xt.append(nats2bits * binxm)
                    #         running_mi_ty.append(nats2bits * binym)
                    #
                    #     self.running_mis_xt[phase].append(running_mi_xt)
                    #     self.running_mis_ty[phase].append(running_mi_ty)

                to_print += f'{phase}: loss {running_loss:>.4f} '  # - acc {acc:>.4f} \t'
                self.losses[phase].append(running_loss)
            print(f'Epoch {i:>4}: {to_print}')

    def plot_losses(self):
        plt.figure()
        for phase in ['train', 'test']:
            plt.plot(self.losses[phase], label=phase)
        plt.legend()
        plt.show()

    def plot_info_plan(self, phase):
        plt.figure()
        plt.title(phase)
        plt.plot(self.running_mis_xt[phase])
        plt.plot(self.running_mis_ty[phase])
        plt.show()

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