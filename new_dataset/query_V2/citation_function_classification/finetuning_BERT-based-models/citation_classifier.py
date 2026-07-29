from torch import nn
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

from torch import nn
import torch

class CitationClassifier(nn.Module):
    """
    Extract CITSEG representation then do classification.
    Optionally concatenate:
        - learned section embedding
        - or numeric position embedding [section_from_beginning, sections_left]
    """

    def __init__(self, linear_size, model, tokenizer, in_features, num_class, use_section=False, num_sections=None, section_emb_dim=32, dropout=0.5, max_sections=7):
        super(CitationClassifier, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.in_features = in_features
        self.use_section = use_section
        self.section_emb_dim = section_emb_dim
        self.max_sections = max_sections
        self.use_position_embedding = self.use_section and self.section_emb_dim == "Position_embedding"

        if self.use_section:
            if self.use_position_embedding:
                self.section_embedding = None
                classifier_input_size = in_features + 2
            else:
                if num_sections is None:
                    raise ValueError("num_sections must be provided when use_section=True")
                self.section_embedding = nn.Embedding(num_embeddings=num_sections, embedding_dim=section_emb_dim)
                classifier_input_size = in_features + section_emb_dim
        else:
            self.section_embedding = None
            classifier_input_size = in_features

        self.linear1 = nn.Linear(in_features=classifier_input_size, out_features=linear_size)
        self.norm1 = nn.LayerNorm(linear_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=linear_size, out_features=num_class)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, tokens, attention_mask, section_ids=None):
        bert_output = self.model(input_ids=tokens, attention_mask=attention_mask)
        last_hidden_state = bert_output.last_hidden_state
        batch_size = tokens.shape[0]

        citseg_id = self.tokenizer.convert_tokens_to_ids("CITSEG")
        citseg_mask = tokens.eq(citseg_id)

        if not citseg_mask.any(dim=1).all():
            raise ValueError("At least one sequence does not contain CITSEG.")

        citseg_positions = citseg_mask.float().argmax(dim=1).long()
        batch_indices = torch.arange(batch_size, device=tokens.device)
        citseg_embeddings = last_hidden_state[batch_indices, citseg_positions, :]

        if self.use_section:
            if section_ids is None:
                raise ValueError("section_ids must be provided when use_section=True")

            if self.use_position_embedding:
                section_vec = section_ids.float()
                if section_vec.dim() != 2 or section_vec.shape[1] != 2:
                    raise ValueError("For Position_embedding, section_ids must have shape [batch_size, 2].")
                section_vec = torch.clamp(section_vec, min=0, max=self.max_sections)
                section_vec = section_vec / float(self.max_sections)
            else:
                if section_ids.dim() > 1:
                    section_ids = section_ids.squeeze(-1)
                section_vec = self.section_embedding(section_ids.long())

            x = torch.cat([citseg_embeddings, section_vec], dim=1)
        else:
            x = citseg_embeddings

        x = self.linear1(x)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        return x
    

    
def eval_prediction(y_batch_actual, y_batch_predicted, all_labels):
    """Return batches of accuracy, f1 scores and P, R, F per class."""

    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = torch.argmax(y_batch_predicted, dim=1).cpu().detach().numpy()
    
    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')
    f1_macro = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='macro')

    precision_per_class = precision_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)
    recall_per_class = recall_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)
    f1_per_class = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average=None, labels=range(len(all_labels)), zero_division=0)

    return acc, f1, f1_macro, precision_per_class, recall_per_class, f1_per_class

        
def training_step(dataloader, model, optimizer, loss_fn, device, ACC_STEP, scheduler):
    model.train()
    epoch_loss = 0.0

    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        section_ids = batch.get("section_ids")
        if section_ids is not None:
            section_ids = section_ids.to(device)

        outputs = model(tokens=input_ids, attention_mask=attention_mask, section_ids=section_ids)

        loss = loss_fn(outputs, labels)
        epoch_loss += loss.item()

        loss = loss / ACC_STEP
        loss.backward()

        if (i + 1) % ACC_STEP == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return epoch_loss / len(dataloader)

def validation_step(dataloader, model, loss_fn, device, all_labels):        
    model.eval()
    
    size = len(dataloader)
    f1, acc, total_loss, f1_macro_total = 0, 0, 0, 0

    all_precision_per_class = []
    all_recall_per_class = []
    all_f1_per_class = []
    
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)
                
            section_ids = batch.get("section_ids")
            if section_ids is not None:
                section_ids = section_ids.to(device)

            pred = model(tokens=X, attention_mask=attention_mask, section_ids=section_ids)

            loss = loss_fn(pred, y)  
            total_loss += loss.item()
            
            acc_batch, f1_batch, f1_macro_batch, precision_per_class, recall_per_class, f1_per_class = eval_prediction(y.float(), pred, all_labels)                        
            acc += acc_batch
            f1 += f1_batch
            f1_macro_total += f1_macro_batch


            all_precision_per_class.append(precision_per_class)
            all_recall_per_class.append(recall_per_class)
            all_f1_per_class.append(f1_per_class)

        acc = acc/size
        f1 = f1/size
        f1_macro_total = f1_macro_total / size 

        max_classes = max(map(len, all_precision_per_class))

        all_precision_per_class = [np.pad(p, (0, max_classes - len(p)), 'constant', constant_values=np.nan) for p in all_precision_per_class]
        all_recall_per_class = [np.pad(r, (0, max_classes - len(r)), 'constant', constant_values=np.nan) for r in all_recall_per_class]
        all_f1_per_class = [np.pad(f, (0, max_classes - len(f)), 'constant', constant_values=np.nan) for f in all_f1_per_class]

        precision_per_class = np.nanmean(all_precision_per_class, axis=0)
        recall_per_class = np.nanmean(all_recall_per_class, axis=0)
        f1_per_class = np.nanmean(all_f1_per_class, axis=0)

                
    return acc, f1, total_loss, f1_macro_total, precision_per_class, recall_per_class, f1_per_class