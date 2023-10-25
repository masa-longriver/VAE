import datetime as dt
import os
import matplotlib.pyplot as plt
import torch

def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
       if i == 0:
          continue
       else:
          now_path += f"/{dir}"
          if not os.path.exists(now_path):
             os.makedirs(now_path)


class EarlyStopping():
    def __init__(self, config, dataset_nm):
        self.config = config
        self.dataset_nm = dataset_nm

        self.path = os.path.join(os.getcwd(),
                                 'models',
                                 self.dataset_nm)
        make_dir(self.path)
        self.best_loss = float('inf')
        self.patience = 0
    
    def save_model(self, model):
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_nm = os.path.join(self.path, f"{now}_VAE")
        torch.save(model.state_dict, file_nm)
    
    def check(self, loss, model, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0

            if epoch + 1 == self.config['epochs']:
                print("========== All epochs are end. ==========", flush=True)
                print(f"Best valid loss: {self.best_loss:.4f}", flush=True)
                self.save_model(model)

            return False
        
        else:
            self.patience += 1
            
            if self.patience >= self.config['model']['patience']:
                print("========== Early Stopping ==========", flush=True)
                print(f"Best valid loss: {self.best_loss:.4f}", flush=True)
                self.save_model(model)

                return True
            
            if epoch + 1 == self.config['epochs']:
                print("========== All epochs are end. ==========", flush=True)
                print(f"Best valid loss: {self.best_loss:.4f}", flush=True)
                self.save_model(model)
            
            return False


def save_loss(train_loss, valid_loss, dataset):
    path = os.path.join(os.getcwd(), 'log', 'losses', dataset.lower())
    make_dir(path)
    
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title("losses")
    plt.ylabel('loss')
    plt.xlabel("epoch")
    plt.legend()

    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_nm = os.path.join(path, f'{now}_loss.png')
    plt.savefig(file_nm)
    plt.close()

def save_img(config, model, dl, dataset):
    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(os.getcwd(), 'log', 'img', dataset.lower(), now)
    make_dir(path)

    with torch.no_grad():
        for i, (x, _) in enumerate(dl):
            if i >= 20:
                break
            x = x.to(config['device'])
            x_out, _, _ = model(x)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
            axes[1].imshow(x_out[0].permute(1, 2, 0).cpu().numpy())
            plt.savefig(os.path.join(path, f'img_{i+1}.png'))