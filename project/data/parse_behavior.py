import argparse
from os import path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import random

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='behaviour file', required=True)
parser.add_argument('--out-dir', action='store', dest='out_dir',
                    help='parsed/pre-processed behaviour file dir', required=True)
parser.add_argument('--mode', action='store', dest='mode',
                    help='train or test', required=True)
parser.add_argument('--user2int', action='store', dest='user2int',
                    help='user index map')
parser.add_argument('--split', action='store', dest='split',
                    help='train/val split', default=0.1)
parser.add_argument('---n-negative-samples', action='store', dest='n_negative',
                    help='number of negative samples per positive sample', default=4)

args = parser.parse_args()


# helper to load map (e.g. user-index) as dict
def load_idx_map_as_dict(file_name):
    with open(file_name, 'r') as file:
        dictionary = {}
        lines = file.readlines()
        for line in tqdm(lines):
            key, value = line.strip().split("\t")
            dictionary[key] = value
        return dictionary

# for each positive candidate sample N negative canidates
# <uid>,<news ids of clickhistory>,<pos. candidate id + N negative candidate ids><mask: 1 + 0*N>
def generate_training_data(behavior, out_dir):
    print("preparing training data")
    random.seed(1234)
    with open(path.join(out_dir, "train_behavior.tsv"), 'w') as train_out:
        train_writer = csv.writer(train_out, delimiter='\t')
        user2int = {}
        for b in tqdm(behavior): 
            imp_id, userid, imp_time, click, imps = b.strip().split("\t")
            if userid not in user2int:
                user2int[userid] = len(user2int) + 1
            positive = [x[:-2] for x in imps.strip().split(" ") if x.endswith("1")]
            negative = [x[:-2] for x in imps.strip().split(" ") if x.endswith("0")]
            if (len(positive) < 1 or len(negative) < args.n_negative):
                continue
            for p in positive:
                ns  = random.sample(negative, args.n_negative)
                pair = " ".join([p] + ns)
                mask = " ".join(["1"]+["0"]*args.n_negative)
                out = [user2int[userid], click, pair, mask]
                train_writer.writerow(out)
        with open(path.join(out_dir, 'user2int.tsv'), 'w') as file:  
            user_writer = csv.writer(file, delimiter='\t')
            for key, value in user2int.items():
                user_writer.writerow([key, value])
            return user2int
# eval data is not balanced
# <uid>,<news ids of clickhistory>,<candidate ids><click mask>
def generate_eval_data(behavior, out_dir, out_file_name, user2int):
    print("preparing eval data")
    with open(path.join(out_dir, out_file_name), 'w') as eval_out:
        eval_writer = csv.writer(eval_out, delimiter='\t')
        for b in tqdm(behavior): 
            imp_id, userid, imp_time, click, imps = b.strip().split("\t")
            impressions =  " ".join([x[:-2] for x in imps.strip().split(" ")])
            impressions_mask = " ".join(["1" if x.endswith('1') else "0" for x in imps.strip().split(" ")])
            out = [user2int.get(userid, 0), click, impressions, impressions_mask]
            eval_writer.writerow(out)

if(args.mode == "train"):
    with open(args.in_file, 'r') as in_file:
        behavior = in_file.readlines()
        if (args.split == 0):
            generate_training_data(behavior, args.out_dir)
        else:
            train_behavior, val_behavior = train_test_split(behavior,test_size=args.split, random_state=1234)
            user2int = generate_training_data(train_behavior, args.out_dir)
            generate_eval_data(val_behavior, args.out_dir, "val_behavior.tsv", user2int)
elif(args.mode == "test"):
    user2int = load_idx_map_as_dict(args.user2int)
    with open(args.in_file, 'r') as in_file:
        behavior = in_file.readlines()
        generate_eval_data(behavior, args.out_dir, "test_behavior.tsv", user2int)
else:
    print("Wrong mode!")