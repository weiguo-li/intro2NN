from torch import nn
import torch



# use Bradley Terry to model the preference 

class OutcomeRewardModel(nn.Module):
    def __init__(self, model_backbone, *args, **kwargs) -> None:
        super().__init__()
        self.backbone = model_backbone
        self.reward_head = nn.Linear(model_backbone.config.hidden_size, 1, device = model_backbone.device)
        self.device = model_backbone.device



    def forward(self, input_ids, attention_mask = None):
        output = self.backbone(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)
        last_layer_hidden_states = output.hidden_states[-1]

        if attention_mask is not None:
            seq_len = attention_mask.sum(dim = -1) - 1 # this operation will reduce the dimension 
            last_token_hidden_states = last_layer_hidden_states[torch.arange(last_layer_hidden_states.size(0)), seq_len] # Size(batch, hidden_dim)
        else:
            last_token_hidden_states = last_layer_hidden_states[:, -1, :] 

        reward = self.reward_head(last_token_hidden_states).squeeze(-1)
        return reward
        

        