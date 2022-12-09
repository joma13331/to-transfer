from pathlib import Path

from transformers import BertTokenizer
import torch
import bentoml

model_state_dict_path = "../model/pytorch_model.bin"

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertModel, BertPreTrainedModel



class BertForMultiLabelSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, pos_weight=None):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        print(f"outputs: {outputs}")

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        print(f"logits: {logits}")
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
            labels = labels.float()
            print(logits, labels)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            outputs = (loss, ) + outputs

        return outputs

def save_pytorch_model():
    model_state_dict = torch.load(model_state_dict_path, map_location=torch.device('cpu'))
    model = BertForMultiLabelSequenceClassification.from_pretrained("../model", num_labels=5,
                                                                    state_dict=model_state_dict)
    model.eval()
    torch.save(model, "../st_pytorch_model.pt")


def load_model_save_bento_model(model_path: Path) -> None:
    model = torch.load(model_path)
    model.eval()
    bento_model = bentoml.pytorch.save_model("st_pytorch_model", model,
                                             signatures={"__call__": {"batchable": True, "batch_dim": 0}})
    print(f"Bento model tag = {bento_model.tag}")
    tokenizer = BertTokenizer.from_pretrained("../model", do_lower_case=False)
    pickle_artifact = bentoml.picklable_model.save_model("st_tokenizer", tokenizer)
    print(f"Pickle Artifact tag = {pickle_artifact.tag}")


if __name__ == "__main__":
    save_pytorch_model()
    load_model_save_bento_model(Path("../st_pytorch_model.pt"))
