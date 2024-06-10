import argparse
import numpy as np
import os
import sys
import tqdm
import torch
import torchinfo
import logging
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from local.training import SequentialSpeechTrials, StoreBestModel, AsynchronousSynthesisQueue
from local.models import BidirectionalSpeechSynthesisModel
from local.common import SelectElectrodesOverSpeechAreas, LeaveOneDayOut
from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    # Parameters
    nb_hidden_units: int
    nb_layer: int
    nb_epochs: int
    batch_size: int
    num_workers: int

    # Folder paths
    speech_corpus_root: Path
    out_dir: Path
    test_day: str
    valid_day: str


def main(train_config: TrainingConfiguration):
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
        ])

    logger = logging.getLogger("train_nVAD")

    E = len(SelectElectrodesOverSpeechAreas())   # Number of features
    logger.info(f"Number of channels: {E}, {SelectElectrodesOverSpeechAreas()}")

    summary_writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))
    best_model = StoreBestModel(filename=os.path.join(out_dir, "best_model.pth"))

    feature_files = list(Path(train_config.speech_corpus_root).rglob('KeywordReading_Overt_R*.hdf'))
    groups_by_day = defaultdict(list)
    for feature_file in feature_files:
        day = feature_file.parent.name
        groups_by_day[day].append(feature_file)

    os.makedirs(os.path.join(out_dir, "orig"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reco"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)

    kf = LeaveOneDayOut()
    syn_queue = AsynchronousSynthesisQueue(nb_processes=8)
    synthesized_orig = False
    for (train_days, test_day) in kf.split(X=groups_by_day.keys(), start_with_day=train_config.test_day):
        kf_va = LeaveOneDayOut()
        train_days, val_day = next(kf_va.split(train_days, start_with_day=train_config.valid_day))
        logger.info(f"Starting Leave-one-day-out cross validation with {test_day} as test and "
                    f"{val_day} as validation day")

        tr_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name in train_days]
        va_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name == val_day]
        tr_files = [f for f in tr_files if f not in va_files]
        te_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name == test_day]
        te_files = sorted(te_files)

        # Initialize datasets
        tr_dataset = SequentialSpeechTrials(feature_files=tr_files, transform=SelectElectrodesOverSpeechAreas())
        va_dataset = SequentialSpeechTrials(feature_files=va_files, transform=SelectElectrodesOverSpeechAreas())
        te_dataset = SequentialSpeechTrials(feature_files=te_files, transform=SelectElectrodesOverSpeechAreas())

        # Initialize the dataloader for all three datasets
        dataloader_params = dict(batch_size=train_config.batch_size, num_workers=train_config.num_workers,
                                 pin_memory=True)
        tr_dataloader = DataLoader(tr_dataset, **dataloader_params, shuffle=True)
        va_dataloader = DataLoader(va_dataset, **dataloader_params, shuffle=True)
        te_dataloader = DataLoader(te_dataset, **dataloader_params, shuffle=False)
        tr_dataloader_unshuffled = DataLoader(tr_dataset, batch_size=train_config.batch_size, shuffle=False,
                                              num_workers=train_config.num_workers, pin_memory=True)

        # Prepare the decoding model that is going to be trained
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Setting device to: {device}')

        model = BidirectionalSpeechSynthesisModel(nb_layer=train_config.nb_layer,
                                                  nb_hidden_units=train_config.nb_hidden_units,
                                                  nb_electrodes=E,
                                                  dropout=0.5)

        net_name = type(model).__name__
        optim = torch.optim.RMSprop(model.parameters(), lr=0.0001)
        cfunc = nn.MSELoss(reduction='mean')

        nb_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters of the {net_name} model: {nb_train_params:,}')

        model.to(device)
        summary = torchinfo.summary(model, input_size=[(train_config.batch_size, 100, E)],
                                    row_settings=['var_names'], col_names=['input_size', 'output_size', 'num_params'],
                                    verbose=1)

        summary_writer.add_graph(model=model,
                                 input_to_model=[
                                     torch.zeros(train_config.batch_size, 100, E).to(device).float(),
                                     model.create_new_initial_state(batch_size=1, device=device)
                                 ])
        summary_writer.flush()
        with open(os.path.join(out_dir, 'model.network'), 'w+') as f:
            f.write(str(summary))

        # Train the model for the specified fold
        for epoch in range(train_config.nb_epochs):
            # Keep running loss during epoch computation
            train_loss = 0
            valid_loss = 0
            seen_minibatch_counter = 0

            # Iterate over all trials
            model.train()
            pbar = tqdm.tqdm(tr_dataloader, total=len(tr_dataloader))
            for x_train, y_train in tr_dataloader:
                # Initialize state
                init_state = model.create_new_initial_state(batch_size=x_train.size(0), device=device, req_grad=True)

                # Process each sample in the current trial
                x_train = x_train.to(device=device).float()
                y_train = y_train.to(device=device).float()

                # Set gradient memory to None
                for param in model.parameters():
                    param.grad = None

                # Forward pass
                pred, _ = model(x_train, state=init_state)
                loss = cfunc(pred, y_train)

                # Backward pass
                loss.backward()
                optim.step()

                # Log running loss on training data
                train_loss += loss.item()
                seen_minibatch_counter += x_train.size(0)

                pbar.set_description(f'Epoch {epoch + 1:>04}: Train loss: {train_loss / seen_minibatch_counter:.04f} '
                                     f'-- Validation loss:...')
                pbar.update()

            # Iterate over validation data for evaluation
            final_train_loss = train_loss / seen_minibatch_counter
            seen_minibatch_counter = 0
            model.eval()
            for x_val, y_val in va_dataloader:
                init_state = model.create_new_initial_state(batch_size=x_val.size(0), device=device)

                # Process each sample in the current trial
                x_val = x_val.to(device=device).float()
                y_val = y_val.to(device=device).float()

                # Make model predictions
                output, _ = model(x_val, init_state)

                # Compute loss
                loss = cfunc(output, y_val)
                valid_loss += loss.item()
                seen_minibatch_counter += x_val.size(0)

            # Update progress bar for current epoch with validation loss
            pbar.set_description(f'Epoch {epoch + 1:>04}: Train loss: {final_train_loss:.04f} '
                                 f'-- Validation loss: {valid_loss / seen_minibatch_counter:.04f}')
            pbar.update()
            pbar.close()
            logger.info(f'Epoch {epoch + 1:>04}: Train loss: {final_train_loss:.04f} '
                        f'-- Validation loss: {valid_loss / seen_minibatch_counter:.04f}')
            final_valid_loss = valid_loss / seen_minibatch_counter
            summary_writer.add_scalars("Training vs. validation loss",
                                       {"Train": final_train_loss, "Valid": final_valid_loss}, epoch + 1)
            best_model.update(model=model, validation_loss=final_valid_loss)

            # Synthesize validation sample
            model.eval()
            test_sentences = list()
            orig_sentences = list()
            for i, (x_test, y_test) in enumerate(te_dataloader):
                if i == 30:
                    break

                x_test = x_test.to(device=device).float()
                init_state = model.create_new_initial_state(batch_size=x_test.size(0), device=device)
                output, _ = model(x_test, init_state)

                test_sentences.append(torch.squeeze(output).cpu().detach().numpy())
                orig_sentences.append(torch.squeeze(y_test).cpu().detach().numpy())

            # Synthesize training samples
            model.eval()
            train_sentences = list()
            orig_train_sentences = list()
            for i, (x_train, y_train) in enumerate(tr_dataloader_unshuffled):
                if i == 30:
                    break

                x_train = x_train.to(device=device).float()
                init_state = model.create_new_initial_state(batch_size=x_train.size(0), device=device)
                output, _ = model.forward(x_train, init_state)

                train_sentences.append(torch.squeeze(output).cpu().detach().numpy())
                orig_train_sentences.append(torch.squeeze(y_train).cpu().detach().numpy())

            # Add results to the asynchronous synthesis queue for being transformed into acoustic waveform
            if not synthesized_orig:
                synthesized_orig = True
                orig_sentences = np.vstack(orig_sentences)
                orig_filename = os.path.join(out_dir, "orig", f"orig.npy")
                np.save(orig_filename, orig_sentences)
                syn_queue.add_job(filename=orig_filename, verbose=0)

                orig_train_sentences = np.vstack(orig_train_sentences)
                orig_train_filename = os.path.join(out_dir, "orig", f"orig_train.npy")
                np.save(orig_train_filename, orig_train_sentences)
                syn_queue.add_job(filename=orig_train_filename, verbose=0)

            test_sentences = np.vstack(test_sentences)
            reco_filename = os.path.join(out_dir, "reco", f"reco_epoch={epoch + 1:03d}.npy")
            np.save(reco_filename, test_sentences)
            syn_queue.add_job(filename=reco_filename, verbose=0)

            train_sentences = np.vstack(train_sentences)
            train_filename = os.path.join(out_dir, "train", f"train_epoch={epoch + 1:03d}.npy")
            np.save(train_filename, train_sentences)
            syn_queue.add_job(filename=train_filename, verbose=0)

        syn_queue.wait()
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the bidirectional speech decoding model.")
    parser.add_argument("corpus_dir", help="Path to the preprocessed corpus with the .hdf files.")
    parser.add_argument("out_dir", help="Path to directory in which the model training will be saved.")
    parser.add_argument("--test_day", help="Day used as offline test data.", default="2022_11_04")
    parser.add_argument("--val_day", help="Day used as validation data.", default="2022_11_03")
    parser.add_argument("--epochs", help="Number of training epochs.", default="100")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # Specify training configuration
    train_config = TrainingConfiguration(
        nb_hidden_units=100,
        nb_layer=2,
        nb_epochs=int(args.epochs),
        batch_size=1,
        num_workers=4,
        speech_corpus_root=Path(args.corpus_dir),
        out_dir=out_dir,
        test_day=args.test_day,
        valid_day=args.val_day
    )

    # Logging functionality
    os.makedirs(out_dir.as_posix(), exist_ok=True)
    log_file = os.path.join(out_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler(sys.stderr)
        ])

    # Train speech decoding model
    main(train_config)
