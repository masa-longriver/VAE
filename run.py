import torch

from model import loss_fn

def run(config, model, dl, optim, state='train'):
    running_loss = 0.0
    if state == 'train':
        model.train()
        for x, _ in dl:
            x = x.to(config['device'])
            optim.zero_grad()
            x_out, mu, logvar = model(x)
            loss = loss_fn(x_out, x, mu, logvar)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
    
    elif state == 'eval':
        model.eval()
        with torch.no_grad():
            for x, _ in dl:
                x = x.to(config['device'])
                x_out, mu, logvar = model(x)
                loss = loss_fn(x_out, x, mu, logvar)
                running_loss += loss.item() * x.size(0)
    
    return running_loss / len(dl)