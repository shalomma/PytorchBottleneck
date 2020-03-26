import pickle
import torch
import simplebinmi


class TrainConfig:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler


class Train:
    def __init__(self, config):
        self.config = config
        self.epochs = 5000
        self.mi_cycle = 10
        self.n_layers = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.losses = dict()
        self.accuracy = dict()
        self.running_mis_xt = dict()
        self.running_mis_ty = dict()
        for phase in ['train', 'test']:
            self.losses[phase] = []
            self.accuracy[phase] = []
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
        self.n_layers = self.config.model.n_layers
        for i in range(self.epochs):
            to_print = ''
            for phase in ['train', 'test']:
                phase_loss = 0.0
                phase_labels = torch.tensor([], dtype=torch.long).to(self.device)
                phase_outputs = torch.tensor([]).to(self.device)

                for inputs, labels in loader[phase]:
                    inputs, labels = inputs, labels
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
                    for j in range(self.n_layers):
                        activity = hiddens[j].cpu().detach().numpy()
                        binxm, binym = simplebinmi.bin_calc_information(class_masks[phase], activity, binsize=0.5)
                        running_mi_xt.append(binxm)
                        running_mi_ty.append(binym)

                    self.running_mis_xt[phase].append(running_mi_xt)
                    self.running_mis_ty[phase].append(running_mi_ty)

                n = float(len(loader[phase].dataset))
                loss = phase_loss / n
                acc = (phase_labels == phase_outputs.argmax(dim=1)).sum() / n
                self.accuracy[phase].append(acc)

                to_print += f'{phase}: loss {loss:>.4f} - acc {acc:>.4f} \t'
                self.losses[phase].append(loss)
                if phase == 'test':
                    if self.config.scheduler is not None:
                        self.config.scheduler.step(loss)
            print(f'Epoch {i:>4}: {to_print}')

    def dump(self):
        tracking = {
            'n_layers': self.n_layers,
            'mi_cycle': self.mi_cycle,
            'losses': self.losses,
            'accuracy': self.accuracy,
            'running_mis_xt': self.running_mis_xt,
            'running_mis_ty': self.running_mis_ty,
        }
        with open('train.pkl', 'wb') as f:
            pickle.dump(tracking, f)
