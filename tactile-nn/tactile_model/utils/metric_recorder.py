class MetricRecorder:
    def __init__(self, name, verbose):
        self.name = name
        self.verbose = verbose

        # metrics to record
        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_samples = 0

    def reset(self):
        # metrics to record
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_acc = 0

    def update(self, loss, num_samples, acc=None):
        if loss is not None:
            self.total_loss += loss.cpu().detach().item() * num_samples
        self.total_samples += num_samples
        if acc is not None:
            self.total_acc += acc.cpu().detach().item() * num_samples

    def fetch_and_print(self, epoch=None, lr=None):
        avr_loss = self.total_loss / self.total_samples
        avr_acc = self.total_acc / self.total_samples

        if self.verbose:
            print()
            print(f'{self.name} | mean loss {avr_loss:5.2f} | mean acc {avr_acc:5.2f} | lr {lr} | epoch {epoch}')
        return {'avr_loss': avr_loss, 'avr_acc': avr_acc, }
