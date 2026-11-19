"""Create a patient-disjoint internal split without using outcome scores."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from split_protocol import patient_id, read_cases


def choose_internal_cases(development, test, seed, validation_cases=100):
    if len(development) != len(set(development)) or len(test) != len(set(test)):
        raise ValueError('Duplicate case IDs.')
    if {patient_id(c) for c in development} & {patient_id(c) for c in test}:
        raise ValueError('Development and test patients overlap.')
    groups=defaultdict(list)
    for case in sorted(development):groups[patient_id(case)].append(case)
    singles=sorted(p for p,cases in groups.items() if len(cases)==1)
    doubles=sorted(p for p,cases in groups.items() if len(cases)==2)
    if len(singles)+len(doubles)!=len(groups):
        raise ValueError('This sampler supports one or two examinations per patient.')
    paired_count=round(len(doubles)*validation_cases/len(development))
    single_count=validation_cases-2*paired_count
    if single_count<0 or single_count>len(singles) or paired_count>len(doubles):
        raise ValueError('Requested validation size cannot be sampled with these strata.')
    rng=random.Random(seed)
    selected=set(rng.sample(doubles,paired_count)+rng.sample(singles,single_count))
    val=sorted(case for patient in selected for case in groups[patient])
    train=sorted(set(development)-set(val))
    if not train or len(val)!=validation_cases:raise ValueError('Invalid split size.')
    return train,val


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-list',required=True)
    parser.add_argument('--test-list',required=True)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--validation-cases',default=100,type=int)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    development=read_cases(args.development_list);test=read_cases(args.test_list)
    train,val=choose_internal_cases(development,test,args.seed,args.validation_cases)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    for name,ids in [('development',development),('train',train),('val',val),('test',test)]:
        (out/(name+'.txt')).write_text('\n'.join(ids)+'\n')
    info={'internal_split_seed':args.seed,'purpose':'Generated internal lists for new training runs.',
          'counts':{n:len(ids) for n,ids in [('development',development),('train',train),('val',val),('test',test)]},
          'development_sha256':hashlib.sha256(Path(args.development_list).read_bytes()).hexdigest()}
    (out/'split_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info['counts']))


if __name__=='__main__':main()
