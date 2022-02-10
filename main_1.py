from comet_ml import Experiment
import torch
import os
import torchaudio
import torch.utils.data as data
from payan.Speech_Recognition.SpeechRecognitionModel_1 import SpeechRecognitionModel
import torch.nn as nn
import torch.optim as optim


def main(learning_rate=5e-4, batch_size=20, epochs=1, train_url="train-clean-100", test_url="test-clean",
         experiment=Experiment(api_key='dummy_key', disabled=True)):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    experiment.log_parameters(hparams)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x, 'train'),
                                   **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, 'valid'),
                                  **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    # Model Initialization
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader)),
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    iter_meter = IterMeter()
    FILE = "C:/DeepSpeech2_model2_1.t7"
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
        torch.save({'epoch': epochs, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), }, FILE)
        test(model, device, test_loader, criterion, epoch, iter_meter, experiment)


if __name__ == '__main__':
    # Setting up comet animation and training hyperparameters
    comet_api_key = ""  # add your api key here if not please design your own visualization procedure
    project_name = "speechrecognition"
    experiment_name = "speechrecognition"

    if comet_api_key:
        experiment = Experiment(api_key=comet_api_key, project_name=project_name, log_code=True, parse_args=False)
        experiment.set_name(experiment_name)
    else:
        experiment = Experiment(api_key='dummy_key', disabled=True)

    learning_rate = 1e-3
    batch_size = 10
    epochs = 1
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"

    main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set, experiment)

