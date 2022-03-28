import torch
import torch.nn.init as init
import torch.nn as nn
import matplotlib.pyplot as plt

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)

def save_model(filename, model, optimizer, scheduler, epoch, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist, early_stop_counter):
    """
        Function to save model.
        
        Function saves model and other training related information so that it can be loaded later to resume training or for inference.
        It is called by fit() function to save best model during training.
    """
    state_dict = {
        'epoch':epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_tr_hist': loss_tr_hist,
        'loss_val_hist': loss_val_hist,
        'accuracy_tr_hist': accuracy_tr_hist,
        'accuracy_val_hist': accuracy_val_hist,
        'early_stop_counter': early_stop_counter
    }
    torch.save(state_dict, filename)

def load_model(filename, model, optimizer = None, scheduler = None, mode = 'test'):
    """
        This function loads previously saved model and its related training details from file specified by filename.
        
        Parameters:
            filename : path of saved model file.
            model : Instance of model to be loaded.
            optimizer : Instance of optimizer to be loaded to previous saved state. Useful to resume training of model from saved state.
            scheduler : Instance of scheduler to be loaded to previous saved state. Useful to resume training of model from saved state.
            mode : Values should be 'train' or 'test'. If value is 'train', it returns model and all other information required to resume training from saved state.
                   If value is 'test', it loads and returns only model.
    """
    state_dict = torch.load(filename)

    model.load_state_dict(state_dict['model'])
    if mode == 'test':
        return model

    epoch = state_dict['epoch']
    optimizer.load_state_dict(state_dict['optimizer'])
    loss_tr_hist = state_dict['loss_tr_hist']
    loss_val_hist = state_dict['loss_val_hist']
    accuracy_tr_hist = state_dict['accuracy_tr_hist']
    accuracy_val_hist = state_dict['accuracy_val_hist']
    early_stop_counter = state_dict['early_stop_counter']
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])

    return epoch, model, optimizer, scheduler, early_stop_counter, loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist

def plot(loss_tr_hist, loss_val_hist, accuracy_tr_hist, accuracy_val_hist):
    """ Plots training loss vs validation loss and training accuracy vs validation accuracy graphs. """
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.subplot(121)
    plt.plot(loss_tr_hist)
    plt.plot(loss_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('Training', 'Validation'))

    plt.subplot(122)
    plt.plot(accuracy_tr_hist)
    plt.plot(accuracy_val_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(('Training', 'Validation'))
    plt.show()