import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import pad_ids, truncate_sequences
from itertools import chain
from tqdm import tqdm
import numpy as np
from os.path import join
import pickle

SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "[PAD]",
    "additional_special_tokens": ["[SYS]", "[USR]", "[KG]","[SUB]", "[PRED]","[OBJ]","[TRIPLE]", "[SEP]","[Q]"],
}

SPECIAL_TOKENS_VALUES = ["[BOS]", "[EOS]", "[PAD]", "[USR]", "[SUB]", "[SYS]", "[USR]", "[SUB]", "[PRED]","[OBJ]","[TRIPLE]", "[SEP]","[Q]"]

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.sys_token, self.usr_token, self.kg, self.sub_token, self.pred_token, self.obj_token, self.triple_token, self.sep,self.ques = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["additional_special_tokens"])
        self.dialogs = self._prepare_conversations(dataset=name, split_type=split_type)

        self._create_examples()

    def build_input_from_segments(self, knowledge, history, response, example, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        sequence = [[self.bos] + knowledge]  + history+ [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [ [self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token] + s
            for i, s in enumerate(sequence[1:])]  # From the history
        hist_token_type = [[self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token]*(len(s)+1)
                for i, s in enumerate(sequence[1:])
        ]
        sequence = [sequence[0]] + sequence_with_speaker
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.kg]*len(sequence[0])+list(chain(*hist_token_type))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["entity_ids"] = [1]*len(instance["lm_labels"]) if len(example["entity_ids"])==0 else example["entity_ids"][:len(instance["lm_labels"])]
        instance["triple_ids"] = [1]*len(instance["lm_labels"]) if len(example["triple_ids"])==0 else example["triple_ids"][:len(instance["lm_labels"])]
        instance["pos_ids"] = [j for j in range(len(instance["input_ids"]))]

        if len(instance["entity_ids"])!=len(instance["input_ids"]):
            instance["entity_ids"]+= [1]*(len(instance["input_ids"])-len(instance["entity_ids"]))
            instance["triple_ids"] += [1] * (len(instance["input_ids"]) - len(instance["triple_ids"]))
        return instance, sequence

    def prepare_attention_matrix(self, kg, weights):
        attention_mat = np.array((5,5))
        return attention_mat


    def _prepare_conversations(self, dataset="incar", split_type="train"):
        print("Loading dialogue data...")
        formatted_dialogs = pickle.load(open(join("data",dataset,split_type+".pkl"),"rb"))
        for d in formatted_dialogs:
            self.prepare_attention_matrix(d["kg"], d["weights"])
        return formatted_dialogs

    def _knowledge_to_sequence(self, kg):
        st=""
        kg_dict = dict()
        for triple in kg:
            if triple[0] not in kg_dict:
                kg_dict[triple[0]] = [triple[1:]]
            else:
                kg_dict[triple[0]].append(triple[1:])
        ent_ids = []
        trip_ids = []
        st_ids = []
        ent_id = [0]
        for kk, vv in kg_dict.items():
            if not st_ids:
                st_ids += self.tokenizer.tokenize('[BOS] [ENT] {}'.format(kk))
            else:
                st_ids+= self.tokenizer.tokenize('[ENT] {}'.format(kk))

            triple_id = [0]*len(st_ids)
            for elem in vv:
                words = self.tokenizer.tokenize('[PRED] {} [OBJ] {}'.format(elem[0],elem[1]))
                st_ids+=words
                triple_id += [triple_id[-1] + 1] * len(words)
            trip_ids+=triple_id
            if not ent_id:
                ent_ids += [0] * len(triple_id)
            else:
                ent_ids += [ent_id[-1] + 1] * len(triple_id)
        return st_ids, ent_ids, trip_ids, kg_dict.copy()

    def get_weighted_triples(self, dialogue):
        ctx_q = set()
        for item in dialogue["weights"]["question-based"]:
            ctx_q.add(item[0])

        ctx_q = list(ctx_q)
        ctx_q.sort(reverse=True)
        # print("TOP RELS: ", top_rels)
        selected_triples = list()
        for ii, val in enumerate(ctx_q[:self.args.top_weights]):
            for jj, w in enumerate(dialogue["weights"]["question-based"]):
                if w[0]==val:
                    selected_triples.append(dialogue["kg"][jj])
        return selected_triples


    def _create_examples(self):
        print("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs):
            dialog_id = dialog["id"]
            ref_ents = dialog["ref_ents"]
            knowledge_text = dialog["kg"] if self.args.top_weights == -1 else self.get_weighted_triples(dialog)
            used_knowledge, ent_ids, trip_ids, kgdict = self._knowledge_to_sequence(knowledge_text)
            used_knowledge = self.tokenizer.convert_tokens_to_ids(used_knowledge)
            used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]

            history = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn)) for turn in dialog["history"]]
            gt_resp = dialog["response"]
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            self.examples.append({
                "history": truncated_history,
                "task": dialog["task"],
                "entity_ids": ent_ids,
                "triple_ids": trip_ids,
                "knowledge": used_knowledge,
                "knowledge_text": knowledge_text,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "dialog_id": dialog_id,
                "reference_entities": ref_ents
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class Dataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(Dataset, self).__init__(args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["knowledge"],example["history"],example["response"], example)
        return instance

    def collate_fn(self, batch):
        maxlen = 1024
        input_ids = [ins["input_ids"] for ins in batch]
        ent_ids = [ins["entity_ids"] for ins in batch]
        triple_ids = [ins["triple_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        pos_ids = [ins["pos_ids"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        ent_ids = torch.tensor(pad_ids(ent_ids, self.pad))
        triple_ids = torch.tensor(pad_ids(triple_ids, self.pad))
        pos_ids = torch.tensor(pad_ids(pos_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        return input_ids, ent_ids, triple_ids, token_type_ids, pos_ids, lm_labels


class EvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(EvalDataset, self).__init__(args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
