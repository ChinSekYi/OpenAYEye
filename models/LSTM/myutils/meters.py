import numpy as np

class Loss_Meter():
    def __init__(self,  loss_type, train_loss = [], val_loss = [], test_loss = []):
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.loss_type = loss_type
    
    def update(self, loss):
        if self.loss_type == 'train':
            self.train_loss.append(loss)
        elif self.loss_type == 'val':
            self.val_loss.append(loss)
        elif self.loss_type == 'test':
            self.test_loss.append(loss)
            
    def mean(self):
        if self.loss_type == 'train':
            return np.mean(self.train_loss)
        elif self.loss_type == 'val':
            return np.mean(self.val_loss)
        elif self.loss_type == 'test':
            return np.mean(self.test_loss)

# meter = Meter()
# meter.update(1.0)
# meter.update(2.0)
# meter.update(3.0)
# print(meter.loss, meter.mean())