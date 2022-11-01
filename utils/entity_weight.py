import numpy as np
import torch
import torch.nn as nn
import random
from collections import OrderedDict
from transformers import RobertaTokenizer, RobertaForMaskedLM

def set_seed(seedval):
    random.seed(seedval)
    np.random.seed(seedval)
    torch.manual_seed(seedval)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seedval)

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs


class NodeWeights:
    def __init__(self):
        self.TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
        self.LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
        self.LM_MODEL.cuda()
        self.LM_MODEL.eval()
        self.soft_fn = nn.Softmax(dim=1)
        print('Roberta-large: loading done')

    def get_LM_score(self, eids, id2e, question, batch_size = 1):
        eids = eids[:]
        #eids.insert(0, -1)  # QAcontext node
        sents, scores = [], []
        for eid in eids:
            sent = '{} {}.'.format(question.lower(), ' '.join(id2e[eid].split('_')))
            sent = self.TOKENIZER.encode(sent, add_special_tokens=True)
            sents.append(sent)
        n_cids = len(eids)
        cur_idx = 0
        while cur_idx < n_cids:
            # Prepare batch
            input_ids = sents[cur_idx: cur_idx + batch_size]
            max_len = max([len(seq) for seq in input_ids])
            for j, seq in enumerate(input_ids):
                seq += [self.TOKENIZER.pad_token_id] * (max_len - len(seq))
                input_ids[j] = seq
            input_ids = torch.tensor(input_ids).cuda()  # [B, seqlen]
            mask = (input_ids != 1).long()  # [B, seq_len]

            # Get LM score
            with torch.no_grad():
                outputs = self.LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
                loss = outputs[0]  # [B, ]
                _scores = list(-loss.detach().cpu().numpy())  # list of float
                #print(_scores)
            scores += _scores
            cur_idx += batch_size
        assert len(sents) == len(scores) == len(eids)
        cid2score = OrderedDict(sorted(list(zip(eids, scores)), key=lambda x: -x[1]))  # score: from high to low
        # print("CID SCORE: ", cid2score)
        # probscores = [max(1.0+v,1e-10) for v in list(cid2score.values())]
        # probscores = OrderedDict(list(zip(eids, probscores)))

        #probscores = self.soft_fn(torch.tensor(list(cid2score.values())).unsqueeze(0))
        #probscores = OrderedDict(list(zip(eids, probscores.squeeze(0).tolist())))
        return cid2score #, probscores


if __name__=="__main__":
    set_seed(42)

    data = {
        1: "coach",
        2: "arsenal",
        3: "is",
        4: "arsene wegner",
        5: "england",
        6: "london"
    }

    nw = NodeWeights()
    nodeloss= nw.get_LM_score(eids=[6], id2e=data, question="who is the coach of arsenal ?", batch_size=1)
    print("node score: ",nodeloss[6])

    nodelosss= nw.get_LM_score(eids=[5,6,1,2], id2e=data, question="is london the capital of england?", batch_size=1)
    #print(nodescore2)
    print(nodelosss)
