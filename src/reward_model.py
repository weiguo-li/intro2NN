import torch.nn as nn 



# use Bradley Terry to model the preference 

class OutcomeRewarModel(nn.Module):
    def __init__(self, model_backbone, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = model_backbone
        self.reward_head = nn.Linear(model.config.hidden_size, 1)


    def forward(self, input_ids, attention_maks = None):
        self.backbone.transformer(input_ids = input_ids, attention_mask = attention_mask)
        pass