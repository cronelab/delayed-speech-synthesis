import argparse
import numpy as np
import os
import torch
import torchinfo
import logging
import sys
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from local.models import UnidirectionalVoiceActivityDetector
from local.training import SequentialSpeechTrials, StoreBestModel
from local.common import SelectElectrodesOverSpeechAreas, LeaveOneDayOut, VoiceActivityDetectionSmoothing
from dataclasses import dataclass


logger = logging.getLogger("train_unidirectional_vad.py")


@dataclass
class TrainingConfiguration:
    # Parameters
    nb_hidden_units: int
    nb_layer: int
    nb_epochs: int
    batch_size: int
    num_workers: int
    truncated_sequence_length: int

    # Folder paths
    speech_corpus_root: Path
    out_dir: Path
    test_day: str
    valid_day: str


def visualize_vad_predictions(pred: np.ndarray, orig: np.ndarray, speech_probs: np.ndarray, filename: Path):
    """
    Plot the original and the predicted curves of the VAD. Title indicates how many frames have been correctly
    classified.
    """
    smoothing = VoiceActivityDetectionSmoothing(nb_features=64, context_frames=5)
    _, smoothed_preds = smoothing.insert(data=np.zeros((len(pred), 64)), speech_labels=pred)

    fig, ax = plt.subplots(1, 1, num=1, clear=True)
    ax.plot(orig, c="black", linestyle="--")
    ax.plot(smoothed_preds, c="orange")
    ax.plot(speech_probs, c="blue")
    ax.axhline(0.5, c="gray", alpha=0.5)
    ax.set_xlim(0, len(speech_probs))
    ax.set_xlabel("Time [seconds]")
    ax.set_ylabel("Probability")
    ax.set_xticks([0, 100])
    ax.set_xticklabels([0, 1])
    ax.set_title(f"Trial accuracy: {list(pred == orig).count(True) / len(pred) * 100:.2f}")
    plt.savefig(filename.as_posix(), dpi=72)


def main(train_config: TrainingConfiguration):
    # Initializing Tensorboard and storage class for the best model
    summary_writer = SummaryWriter(log_dir=os.path.join(train_config.out_dir, "tensorboard"))
    best_model = StoreBestModel(filename=os.path.join(train_config.out_dir, "best_model.pth"))

    # Get all precomputed feature files per day
    feature_files = list(Path(train_config.speech_corpus_root).rglob('*.hdf'))
    groups_by_day = defaultdict(list)
    for feature_file in feature_files:
        day = feature_file.parent.name
        groups_by_day[day].append(feature_file)

    # Prepare training directories
    os.makedirs(os.path.join(out_dir, "valid_viz"), exist_ok=True)

    # Organize recorded data into train, test and validation set. Test and validation set will be set to fixed dates to
    # have one final model that can be used for online testing.
    kf_te = LeaveOneDayOut()
    kf_va = LeaveOneDayOut()

    train_days, test_day = next(kf_te.split(X=groups_by_day.keys(), start_with_day=train_config.test_day))
    train_days, val_day = next(kf_va.split(train_days, start_with_day=train_config.valid_day))

    logger.info(f"Starting Leave-one-day-out cross validation with {test_day} as test and {val_day} as validation day.")

    # Organize feature files in train, test and validation sets
    tr_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name in train_days]
    va_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name == val_day]
    tr_files = [f for f in tr_files if f not in va_files]  # TODO check if needed
    te_files = [feature_file.as_posix() for feature_file in feature_files if feature_file.parent.name == test_day]
    te_files = sorted(te_files)

    # Initialize datasets
    speech_channel_selection = SelectElectrodesOverSpeechAreas()
    tr_dataset = SequentialSpeechTrials(feature_files=tr_files, transform=speech_channel_selection,
                                        target_specifier="vad_labels")
    va_dataset = SequentialSpeechTrials(feature_files=va_files, transform=speech_channel_selection,
                                        target_specifier="vad_labels")
    te_dataset = SequentialSpeechTrials(feature_files=te_files, transform=speech_channel_selection,
                                        target_specifier="vad_labels")

    # Initialize the dataloader for all three datasets
    dataloader_params = dict(batch_size=1, num_workers=train_config.num_workers, pin_memory=True)
    tr_dataloader = DataLoader(tr_dataset, **dataloader_params, shuffle=True)
    va_dataloader = DataLoader(va_dataset, **dataloader_params, shuffle=False)
    te_dataloader = DataLoader(te_dataset, **dataloader_params, shuffle=False)

    # Setup nVAD architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Setting device to: {device}")

    model = UnidirectionalVoiceActivityDetector(
        nb_layer=train_config.nb_layer, nb_hidden_units=train_config.nb_hidden_units,
        nb_electrodes=len(speech_channel_selection), dropout=0.5
    )

    nb_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters of the {type(model).__name__} model: {nb_train_params:,}")

    # Setup nVAD training
    optim = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    cfunc = nn.CrossEntropyLoss()  # Cost function

    model.to(device)
    summary = torchinfo.summary(model, input_size=[(train_config.batch_size, 100, len(speech_channel_selection))],
                                row_settings=['var_names'], col_names=['input_size', 'output_size', 'num_params'],
                                verbose=0)
    summary_writer.flush()
    with open(os.path.join(train_config.out_dir, 'model.network'), 'w+') as f:
        f.write(str(summary))

    # Run truncated backpropagation
    update_steps_counter = 0
    for epoch in range(train_config.nb_epochs):
        # Keep running loss during epoch computation
        train_loss = []

        # Iterate over all trials
        model.train()
        pbar = tqdm.tqdm(tr_dataloader, total=len(tr_dataloader))
        for x_train, y_train in tr_dataloader:
            # Initialize state
            state = model.create_new_initial_state(batch_size=x_train.size(0), device=device)
            state[0].to(device=device).float()
            state[1].to(device=device).float()
            state[0].requires_grad = True
            state[1].requires_grad = True

            # Use truncated backpropagation with k1 == k2
            sequences = zip(x_train.split(train_config.truncated_sequence_length, dim=1),
                            y_train.split(train_config.truncated_sequence_length, dim=1))
            for x_train_seq, y_train_seq in sequences:
                # Zero-out gradients
                for param in model.parameters():
                    param.grad = None

                # Process each sample in the current sequence
                x_train_seq = x_train_seq.to(device=device).float()
                y_train_seq = y_train_seq.to(device=device).float()

                output, state = model(x_train_seq, state)  # Forward propagation

                # Make backward propagation
                loss = cfunc(torch.reshape(output, (-1, 2)), y_train_seq.squeeze().long())
                loss.backward()
                optim.step()
                update_steps_counter += 1

                # Detach state from computational graph
                state = (state[0].detach(), state[1].detach())

                train_loss.append(loss.item())

            pbar.set_description(f'Epoch {epoch + 1:>04}: Train loss: {sum(train_loss) / len(train_loss):.04f} '
                                 f'-- Validation loss:...')
            pbar.update()

        # Compute loss and accuracy on validation data
        model.eval()
        valid_loss = 0
        valid_acc = []
        for val_index, (x_val, y_val) in enumerate(va_dataloader):
            init_state = model.create_new_initial_state(batch_size=x_val.size(0), device=device)

            # Process each sample in the current trial
            x_val = x_val.to(device=device).float()
            y_val = y_val.to(device=device).float()

            output, _ = model(x_val, init_state)  # Forward propagation

            # Compute loss
            loss = cfunc(torch.reshape(output, (-1, 2)), y_val.squeeze().long())
            valid_loss += loss.item()

            # Visualize the output of the neural VAD in comparison with the ground truth
            pred = F.softmax(output, dim=2).argmax(dim=2).squeeze().long().detach().cpu().numpy()
            prob = F.softmax(output, dim=2).squeeze().detach().cpu().numpy()[:, 1]
            orig = y_val.squeeze().long().detach().cpu().numpy()
            epoch_plot_filename = Path(os.path.join(train_config.out_dir, "valid_viz", f"epoch={epoch+1:03d}",
                                                    f"trial_id={val_index:03d}.png"))
            os.makedirs(epoch_plot_filename.parent.as_posix(), exist_ok=True)
            visualize_vad_predictions(pred=pred, orig=orig, speech_probs=prob, filename=epoch_plot_filename)

            # Compute accuracy
            valid_acc.extend(list(pred == orig))

        val_score = valid_acc.count(True) / len(valid_acc)
        pbar.set_description(f'Epoch {epoch + 1:>04}: Train loss: {sum(train_loss) / len(train_loss):.04f} '
                             f'-- Validation loss: {valid_loss:.04f} '
                             f'({update_steps_counter:>6} update steps) [Validation Accuracy: {val_score * 100:.02f}]')
        pbar.update()
        pbar.close()

        # Store new model weights if accuracy score has improved
        best_model.update(model=model, validation_acc=val_score,
                          info={"update_steps": update_steps_counter, "epoch": epoch + 1})

    logger.info(f"Training finished. Best validation accuracy obtained after {best_model.optional_info['update_steps']}"
                f" update steps [epoch {best_model.optional_info['epoch']}].")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the unidirectional VAD model.")
    parser.add_argument("corpus_dir", help="Path to the preprocessed corpus with the .hdf files.")
    parser.add_argument("out_dir", help="Path to directory in which the model training will be saved.")
    parser.add_argument("--test_day", help="Day used as offline test data.", default="2022_11_04")
    parser.add_argument("--val_day", help="Day used as validation data.", default="2022_11_03")
    parser.add_argument("--epochs", help="Number of training epochs.", default="8")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # Specify training configuration
    train_config = TrainingConfiguration(
        nb_hidden_units=150,
        nb_layer=2,
        nb_epochs=int(args.epochs),
        batch_size=1,
        num_workers=0,
        truncated_sequence_length=50,
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

    # Train nVAD model
    main(train_config)
