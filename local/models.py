import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


LstmState = Tuple[torch.Tensor, torch.Tensor]


# region Model architectures
class UnidirectionalVoiceActivityDetector(nn.Module):
    """
    A recurrent neural network that uses unidirectional LSTM cells to predict voice activity in high-gamma frames.
    """
    def __init__(self, nb_layer: int = 2, nb_hidden_units: int = 512, nb_electrodes: int = 128, dropout: float = 0.0):
        super(UnidirectionalVoiceActivityDetector, self).__init__()
        self.nb_hidden_units = nb_hidden_units
        self.nb_layer = nb_layer
        self.lstm = nn.LSTM(input_size=nb_electrodes, hidden_size=self.nb_hidden_units, num_layers=nb_layer,
                            dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(in_features=self.nb_hidden_units, out_features=2)

    def create_new_initial_state(self, batch_size: int, device: str = "cpu", req_grad: bool = False) -> LstmState:
        return (torch.zeros(self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device),
                torch.zeros(self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device))

    def forward(self, x: torch.Tensor, state: Optional[LstmState] = None) -> Tuple[torch.Tensor, LstmState]:
        if state is None:
            state = self.create_new_initial_state(batch_size=x.size(0), device=next(self.parameters()).device)

        x, new_state = self.lstm(x, state)
        out = self.classifier(x)
        return out, new_state


class BidirectionalSpeechSynthesisModel(nn.Module):
    """
    A recurrent neural network that uses bidirectional LSTM cells to synthesize LPC features.
    """
    def __init__(self, nb_layer: int = 2, nb_hidden_units: int = 100, nb_electrodes: int = 128, dropout: float = 0.0):
        super(BidirectionalSpeechSynthesisModel, self).__init__()
        self.nb_hidden_units = nb_hidden_units
        self.nb_layer = nb_layer
        self.lstm = nn.LSTM(input_size=nb_electrodes, hidden_size=self.nb_hidden_units, num_layers=nb_layer,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.regressor = nn.Linear(in_features=(2 * self.nb_hidden_units), out_features=20)

    def create_new_initial_state(self, batch_size: int, device: str = "cpu", req_grad: bool = False) -> LstmState:
        return (torch.zeros(2 * self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device),
                torch.zeros(2 * self.nb_layer, batch_size, self.nb_hidden_units, requires_grad=req_grad, device=device))

    def forward(self, x: torch.Tensor, state: Optional[LstmState] = None) -> Tuple[torch.Tensor, LstmState]:
        if state is None:
            state = self.create_new_initial_state(batch_size=x.size(0), device=next(self.parameters()).device)

        x, new_state = self.lstm(x, state)
        out = self.regressor(x)
        return out, new_state
# endregion
