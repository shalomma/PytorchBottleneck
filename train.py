import torch
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
        self.mi_cycle = 1

        self.losses = dict()
        self.accuracy = dict()
        self.running_mis_xt = dict()
        self.running_mis_ty = dict()
        for phase in ['train', 'test']:
            self.losses[phase] = []
            self.accuracy[phase] = []
            self.running_mis_xt[phase] = []
            self.running_mis_ty[phase] = []

    @staticmethod
    def get_class_masks(data):
        samples_split = dict()
        n_classes = int(data['train']['class'].max()) + 1
        for phase in ['train', 'test']:
            samples_split[phase] = {}
            classes = data[phase]['class'].detach().numpy()
            for i in range(n_classes):
                samples_split[phase][i] = classes == i
        return samples_split

    def run(self, data):
        class_masks = self.get_class_masks(data)

        for i in range(self.epochs):
            to_print = ''
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.config.model.train()
                else:
                    self.config.model.eval()

                self.config.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, hiddens = self.config.model(data[phase]['samples'])
                    loss = self.config.criterion(outputs, data[phase]['labels'])
                    if phase == 'train':
                        loss.backward()
                        self.config.optimizer.step()

                loss = loss.item()
                self.losses[phase].append(loss)
                acc = (data[phase]['class'] == outputs.argmax(dim=1)).sum() / float(len(data[phase]['labels']))
                self.accuracy[phase].append(acc)

                to_print += f'{phase}: loss {loss:>.4f} - acc {acc:>.4f} \t'
                if i % self.mi_cycle == 0:
                    running_mi_xt = []
                    running_mi_ty = []
                    for j in range(len(self.config.model.hidden_sizes)):
                        activity = hiddens[j].detach().numpy()
                        binxm, binym = simplebinmi.bin_calc_information(class_masks[phase], activity, binsize=0.07)
                        running_mi_xt.append(binxm)
                        running_mi_ty.append(binym)

                    self.running_mis_xt[phase].append(running_mi_xt)
                    self.running_mis_ty[phase].append(running_mi_ty)
            print(f'Epoch {i:>4}: {to_print}')
