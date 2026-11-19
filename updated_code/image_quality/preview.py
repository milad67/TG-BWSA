"""Visual QA on an explicitly selected TRAINING case, without loading models."""
import argparse
from pathlib import Path
import json
import numpy as np
from degradation import CONDITIONS, apply_condition, to_chwd
from run_degradation import input_paths, get_foreground, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--case-id", required=True, help="Choose a case from TRAINING for QA, not by its tumor appearance.")
    p.add_argument("--out", default="outputs/qa")
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if not c.get("channel_order_confirmed"):
        raise ValueError("Confirm channel order before checking any transforms.")
    x = to_chwd(np.load(input_paths(c,a.case_id)[0], allow_pickle=False), c["layout"])
    fg, info = get_foreground(c,a.case_id,x)
    z_list = np.flatnonzero(fg.any(axis=(0,1)))
    z = int(z_list[len(z_list)//2])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,5,figsize=(15,6))
    for row, modality in enumerate(("T1ce", "FLAIR")):
        channel = c["channel_order"].index(modality)
        selected = [CONDITIONS[0]] + [q for q in CONDITIONS if q.modality==modality]
        limits = np.percentile(x[channel][fg], [1,99])
        for col, condition in enumerate(selected):
            y = apply_condition(x,fg,condition,c["channel_order"],a.case_id,c["noise_base_seed"])
            assert np.array_equal(y[np.arange(4)!=channel],x[np.arange(4)!=channel])
            axes[row,col].imshow(y[channel,:,:,z].T,cmap="gray",origin="lower",vmin=limits[0],vmax=limits[1])
            axes[row,col].set_title(modality + " / " + condition.name,fontsize=9)
            axes[row,col].axis("off")
    fig.suptitle("Transformation QA only / " + a.case_id)
    fig.tight_layout()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(a.case_id+".png"),dpi=160)
    plt.close(fig)
    save_json(out/(a.case_id+".json"),{"case_id":a.case_id,"z_index":z,"foreground":info,
              "purpose":"Verify transforms on a training case. Do not tune severities using validation scores."})


if __name__ == "__main__":
    main()
