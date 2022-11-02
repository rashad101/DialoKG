import json
import pickle
import ast
from os.path import join
from tqdm import tqdm
from entity_weight import NodeWeights
from relation_relevance import LaplacianMatrix
from gensim.models import fasttext


nw = NodeWeights()
vec = fasttext.load_facebook_vectors("data/wiki.simple.bin")
rw = LaplacianMatrix(vec=vec)

def process_incar():
    splits = ["val","test","train"]
    dataroot = "data/incar"
    for sp in splits:
        with open(join(dataroot,"kvr",sp+".txt")) as f:
            conv_id = 0
            data = dict()
            task = None
            for line in tqdm(f, desc=f"processing files (incar): {sp}:"):
                line = line.strip()
                if line:
                    if '#' in line:
                        line = line.replace("#", "")
                        task = line
                        conv_id += 1
                        data[conv_id] = {
                            "task": task,
                            "utterances": [],
                            "kg": []
                        }
                        continue

                    nid, line = line.split(' ', 1)

                    if '\t' in line:        # conversation
                        u, r, gold_ent = line.split('\t')
                        gold_ent = ast.literal_eval(gold_ent)
                        data[conv_id]["utterances"].append({
                            "user": u,
                            "response": r,
                            "reference_entities": gold_ent
                        })
                    else:                   # kg triples
                        triple = line.split()
                        if task=="weather":
                            if len(triple)==4:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[2]+" "+triple[3]])
                            elif len(triple)==2:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[0]])
                            else:
                                data[conv_id]["kg"].append(triple)
                        else:
                            if len(triple)==3:
                                data[conv_id]["kg"].append(triple)

            json.dump(data, open(join(dataroot,sp+".json"), "w"), indent=3)

def process_kg(dataset, kb):
    triples = list()
    for data in kb:
        ent = data["name"]
        for k,v in data.items():
            if k!="name":
                triples.append([ent,k,v])
    return triples

def process_camrest(dataset):
    splits = ["val","test","train"]
    dataroot = "data/"+dataset

    for sp in splits:
        data = json.load(open(join(dataroot,"raw",sp+".json")))
        save_data = dict()
        for conv_id, item in tqdm(enumerate(data), desc=f"{sp}:"):
            save_data[conv_id] = {
                    "task": item["cusine"],
                    "history": item["context"][:-1],
                    "user": item["context"][-1],
                    "response": item["output"],
                    "reference_entities": item["gold_entities"],
                    "kg": process_kg(dataset=dataset, kb=item["kb"])
            }
        json.dump(save_data, open(join(dataroot,sp+".json"), "w"), indent=3)


def process_woz21(dataset):
    splits = ["val","test","train"]
    dataroot = "data/"+dataset

    for sp in splits:
        data = json.load(open(join(dataroot,"raw",sp+".json")))
        save_data = dict()
        for conv_id, item in tqdm(enumerate(data), desc=f"{sp}:"):
            save_data[conv_id] = {
                    "task": item["type"],
                    "history": item["context"][:-1],
                    "user": item["context"][-1],
                    "response": item["output"],
                    "reference_entities": item["gold_entities"],
                    "kg": process_kg(dataset=dataset, kb=item["kb"])
            }
        json.dump(save_data, open(join(dataroot,sp+".json"), "w"), indent=3)

def weights(text, kg):
    if kg:
        ents = list(set([e[0] for e in kg] + [e[2] for e in kg]))
        emap = {i:e for i,e in enumerate(ents)}
        escores = {v:nw.get_LM_score(eids=[k], id2e=emap, question=text, batch_size=1)[k] for k,v in emap.items()}
        d = dict()
        for triple in kg:
            if triple[0] not in d:
                d[triple[0]] = set([triple[1]])
            else:
                d[triple[0]].add(triple[1])
        hvec, rel_map = rw.relation_relevance(text, d)
        weight_values = [[escores[triple[0]], rel_map[triple[1]], escores[triple[2]]] for triple in kg]
    else:
        return []
    return weight_values

def truncate_long_context(long_text):
    long_text = " ".join(long_text.split()[-400:])
    return long_text

def compute_weights(dataset):
    dataroot = "data/"+dataset
    splits = ["val","test","train"]
    
    for datasplit in splits:
        data = json.load(open(join(dataroot, datasplit+".json")))
        formatted_dialogues = list()
        previous_id = ""
        for dial_id, dg in tqdm(data.items(), desc=f"Computing weights: {dataset}:{datasplit}::: "):  # only show progress bar in one process
            current_id = dial_id
            if dataset=="incar":
                for t,atrun in enumerate(dg["utterances"]):
                    dialog = {}
                    dialog["id"] = dial_id
                    dialog["kg"] = dg["kg"]
                    dialog["task"] = dg["task"]
                    dialog["response"] = atrun["response"]

                    if current_id!=previous_id:
                        dialog["history"] = [atrun["user"]]
                    else:
                        dialog["history"] = formatted_dialogues[-1]["history"] + [dg["utterances"][t-1]["response"],atrun["user"]]

                    dialog["ref_ents"] = atrun["reference_entities"]
                    dialog["weights"] = {
                        "question-based": weights(atrun["user"], kg=dg["kg"]),
                        "context-based": weights(truncate_long_context(long_text=" ".join(h for h in dialog["history"])), kg=dg["kg"])
                    }

                    formatted_dialogues.append(dialog)
                    previous_id=current_id

            elif dataset=="camrest" or dataset=="woz2.1":
                dialog = {}
                dialog["id"] = dial_id
                dialog["kg"] = dg["kg"]
                dialog["task"] = dg["task"]
                dialog["response"] = dg["response"]
                dialog["history"] = dg["history"] + [dg["user"]]
                dialog["ref_ents"] = dg["reference_entities"]

                dialog["weights"] = {
                    "question-based": weights(dg["user"], kg=dg["kg"]),
                    "context-based": weights(truncate_long_context(" ".join(h for h in dialog["history"])), kg=dg["kg"])
                }
                formatted_dialogues.append(dialog)

        pickle.dump(formatted_dialogues, open(join(dataroot, datasplit+".pkl"),"wb"))


def process_entities(dataset):
    if dataset=="camrest" or dataset=="woz2.1":
        ent_data = json.load(open(f"data/{dataset}/raw/entities.json"))
        json.dump(ent_data["all_entities_list"], open(f"data/{dataset}/entities.json","w"), indent=3)


def process_data(dataset="incar"):
    if dataset=="incar":
        process_incar()
        print("Computing weights...... Please wait")
        compute_weights(dataset)
        print("DONE !!")
    elif dataset=="camrest":
        process_camrest(dataset=dataset)
        process_entities(dataset=dataset)
        compute_weights(dataset=dataset)
    elif dataset=="woz2.1":
        process_woz21(dataset=dataset)
        process_entities(dataset=dataset)
        compute_weights(dataset=dataset)


if __name__=="__main__":
    process_data(dataset="incar")
    process_data(dataset="camrest")
    process_data(dataset="woz2.1")
