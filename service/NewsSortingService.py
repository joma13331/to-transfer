import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm.auto import tqdm
from utils import convert_examples_to_features, InputExample


class NewsSortingService:
    args = {
        'threshold': 0.5,
        'num_labels': 5,
        'process_count': 30,
        'max_seq_length': 128,
        'use_multiprocessing': True,
        'eval_batch_size': 8,
        'silent': False
    }

    def load_and_cache_example(self, tokenizer, examples, evaluate=False,
                               multi_label=False, verbose=True, silent=False):

        tokenizer = tokenizer

        output_mode = "classification"

        features = convert_examples_to_features(
            examples,
            self.args['max_seq_length'],
            tokenizer,
            output_mode,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            process_count=self.args['process_count'],
            multi_label=True,
            silent=silent,
            use_multiprocessing=self.args["use_multiprocessing"],
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return dataset

    def get_inputs_dict(self, batch):

        inputs = {"input_ids": batch[0], 'attention_mask': batch[1],
                  'token_type_ids': batch[2], 'labels': batch[3]}
        return inputs

    def threshold(self, x, threshold):

        if x > threshold:
            return 1

        return 0

    def predict(self, to_predict, tokenizer, model):

        multi_label = True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        eval_examples = [InputExample(i, text, [0] * self.args["num_labels"]) for i, text in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_example(tokenizer, eval_examples, evaluate=True,
                                                   multi_label=multi_label)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, disable=self.args['silent']):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            model.to(device)

            with torch.no_grad():
                inputs = self.get_inputs_dict(batch)
                outputs = model(**inputs)

                print(outputs, len(outputs))

                tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        model_outputs = preds
        if multi_label:
            preds = [[self.threshold(pred, self.args["threshold"]) for pred in example] for example in preds]

        else:
            preds = np.argmax(preds, axis=1)

        return preds, model_outputs
