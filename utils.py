import torch
import torch.nn as nn
from transformers import BertModel


class MultiTaskBert(nn.Module):
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_sent_labels: int = 5,
                 num_token_labels: int = 4,
                 pooling: str = "cls"   # for the sentence head
                 ):
        super().__init__()
        # 1) Shared encoder
        self.backbone = BertModel.from_pretrained(model_name)
        hidden_size    = self.backbone.config.hidden_size
        self.pooling   = pooling.lower()

        # 2) Task A: sentence‐level classification head
        self.sent_dropout = nn.Dropout(0.1)
        self.sent_classifier = nn.Linear(hidden_size, num_sent_labels)

        # 3) Task B: token‐level classification head (NER)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(hidden_size, num_token_labels)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        # Shared forward pass
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        token_embs = outputs.last_hidden_state  # (B, T, H)
        cls_emb    = outputs.pooler_output      # (B, H)

        # ——— Task A: Sentence Classification ———
        if self.pooling == "cls":
            sent_repr = cls_emb
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (token_embs * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            sent_repr = summed / lengths
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        sent_logits = self.sent_classifier(self.sent_dropout(sent_repr))
        # shape → (B, num_sent_labels)

        # ——— Task B: Named Entity Recognition ———
        # we apply token‐level dropout, then project each token to NER label space
        token_logits = self.token_classifier(self.token_dropout(token_embs))
        # shape → (B, T, num_token_labels)

        return {
            "sent_logits": sent_logits,
            "token_logits": token_logits
        }
    