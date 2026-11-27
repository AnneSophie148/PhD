import torch

"""
### without identification method
class ArticleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity="relu")
        self.classifier = torch.nn.Linear(hidden_size, 2)
    def forward(self, features):
        out, hidden = self.rnn(features)
        return self.classifier(hidden[-1])

"""

class ArticleRNN(torch.nn.Module):
    def __init__(self, input_size, num_methods, hidden_size=128, num_layers=1):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity="relu")
        self.classifier = torch.nn.Linear(hidden_size + num_methods, 2)
    def forward(self, features, method_features):
        out, hidden = self.rnn(features)
        combined = torch.cat([hidden[-1], method_features.float()], dim=1)
        return self.classifier(combined)