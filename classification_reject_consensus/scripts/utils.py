import torch
from torch.utils.data import Dataset

    
class CitationDataset(Dataset):
    def __init__(self, x_list, labels_ind, num_rhetorical_classes, pooling=False):
        assert len(x_list) == len(labels_ind)
        self.x_list = x_list
        self.labels_ind = labels_ind
        self.num_rhetorical_classes = num_rhetorical_classes
        self.pooling = pooling
    def __len__(self):
        return len(self.x_list)
    def __getitem__(self, idx):
        article = self.x_list[idx]
        article_features = []
        for time_step, citations in sorted(article["features_by_time_step"].items(), key=lambda item: int(item[0])):
            features = []
            for citation in citations:
                if citation["citation_embedding"] is None:
                    continue
                rc = torch.nn.functional.one_hot(torch.tensor(citation["rhetorical_class"], dtype=torch.long), num_classes=self.num_rhetorical_classes).float()
                embedding = torch.tensor(citation["citation_embedding"], dtype=torch.float32)
                features.append(torch.cat([embedding, rc]))
            if features:
                timestep_features = torch.stack(features)
                if self.pooling:
                    timestep_features = timestep_features.mean(dim=0, keepdim=True)
                time_column = torch.full((timestep_features.shape[0], 1), float(time_step), dtype=torch.float32)
                article_features.append(torch.cat([timestep_features, time_column], dim=1))
        if not article_features:
            raise ValueError(f"No valid citations for {article['doi']}")
        
        #return {"doi": article["doi"], "features": torch.cat(article_features, dim=0), "labels": torch.tensor(self.labels_ind[idx], dtype=torch.long)}
        return {
            "doi": article["doi"],
            "features": torch.cat(article_features, dim=0),
            "method_features": torch.tensor(article["method_features"], dtype=torch.float32),
            "labels": torch.tensor(self.labels_ind[idx], dtype=torch.long)
        }