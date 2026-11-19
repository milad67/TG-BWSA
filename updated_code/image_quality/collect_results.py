"""Check complete matched outputs and create a small result archive, no MRI/weights."""
import argparse
import csv
from pathlib import Path
import json
import statistics
import zipfile
from degradation import CONDITIONS, noise_seed
from run_degradation import FIELDS, save_json, read_cases


def collect(config_path):
    c=json.loads(Path(config_path).read_text(encoding="utf-8"))
    root=Path(c["output_dir"])
    expected_cases=read_cases(c["test_file"])
    rows=[]
    first_manifest=None
    summaries=[]
    for model in c["models"]:
        folder=root/model["name"]
        completion=json.loads((folder/"completion.json").read_text())
        meta=json.loads((folder/"run_config.json").read_text())
        if not completion["complete"] or completion["n_conditions"]!=9 or meta["clean_only"]:
            raise ValueError("Incomplete or clean-only run: "+model["name"])
        if meta["cases"]!=expected_cases or meta["config"]!=c:
            raise ValueError("Config or case list mismatch: "+model["name"])
        manifest=meta["input_manifest"]
        if first_manifest is None:
            first_manifest=manifest
        elif manifest!=first_manifest:
            raise ValueError("Models did not use the same input files/versions.")
        if json.loads((folder/"failures.json").read_text()):
            raise ValueError("Failures recorded for "+model["name"])
        by_condition={}
        for cond in CONDITIONS:
            with open(folder/(cond.name+".csv"),newline="",encoding="utf-8") as f:
                records=list(csv.DictReader(f))
            if [r["case_id"] for r in records]!=expected_cases:
                raise ValueError("Missing/duplicate/out-of-order cases: "+str(folder/cond.name))
            for r in records:
                if r["model"]!=model["name"] or r["condition"]!=cond.name or r["checkpoint_sha256"]!=meta["checkpoint_sha256"]:
                    raise ValueError("Unexpected model/condition/checkpoint label.")
                seed=str(noise_seed(r["case_id"],cond.modality,c["noise_base_seed"])) if cond.noise_sigma else ""
                if (r["target_modality"]!=cond.modality or float(r["noise_sigma"])!=cond.noise_sigma
                    or float(r["blur_sigma_vox"])!=cond.blur_sigma_vox or r["noise_seed"]!=seed
                    or r["training_regime"]!="full_input"):
                    raise ValueError("Unexpected corruption parameters, seed or training regime.")
                for key in ("dice_wt","dice_tc","dice_et"):
                    if not 0<=float(r[key])<=1+1e-6:
                        raise ValueError("Invalid Dice value.")
            by_condition[cond.name]=records
            rows.extend(records)
        clean=by_condition["clean"]
        for cond in CONDITIONS:
            records=by_condition[cond.name]
            result={"model":model["name"],"condition":cond.name,"n_cases":len(records)}
            for key in ("dice_wt","dice_tc","dice_et"):
                result[key]=statistics.mean(float(r[key]) for r in records)
                result[key+"_delta_vs_clean"]=statistics.mean(float(r[key])-float(b[key]) for r,b in zip(records,clean))
            summaries.append(result)
    if len(rows)!=len(expected_cases)*len(c["models"])*9:
        raise ValueError("Unexpected final row count.")
    root.mkdir(parents=True,exist_ok=True)
    with open(root/"degradation_per_case.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(root/"degradation_summary.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save_json(root/"degradation_config.json",c)
    (root/"test_cases.txt").write_text("\n".join(expected_cases)+"\n",encoding="utf-8")
    save_json(root/"collection_receipt.json",{"complete":True,"models":len(c["models"]),
               "conditions_per_model":9,"cases_per_condition":len(expected_cases),"rows":len(rows),
               "unit":"Dice 0 to 1; HD95 blank because this wrapper does not compute HD95"})
    archive=root.parent/(root.name+"_results.zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".csv",".json",".txt",".log",".png"):
                z.write(path,"results/"+str(path.relative_to(root)))
        for name in ("degradation.py","run_degradation.py","collect_results.py","preview.py","README.md"):
            path=Path(__file__).with_name(name)
            z.write(path,"code/"+name)
    print("Verified %s rows. Saved %s"%(len(rows),archive.resolve()))
    return archive


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    collect(p.parse_args().config)
