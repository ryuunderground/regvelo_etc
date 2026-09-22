"""Does a ligand atom's embedding know how close that atom sits to the pocket?

BindCLIP's Table 4 probe, run on our checkpoint. A linear model takes each
ligand atom's encoder embedding plus a pooled pocket embedding and predicts that
atom's minimum distance to any pocket atom. Their numbers: RMSE 1.05 for
DrugCLIP, 0.87 for BindCLIP.

The architecture says this should fail. The two encoders never see each other --
the ligand is encoded from its own internal distance matrix, the pocket from
its own, and they meet only at a dot product of two 128-d vectors. No
ligand-atom-to-pocket-atom distance is ever computed. Measuring how badly it
fails is the point: it turns "pose information is absent" from an architectural
argument into a number, and sets the baseline for adding it.

Baselines are reported alongside, because an embedding can look informative
while only encoding atom identity: predicting the mean, and predicting from the
element type alone.
"""
import pickle
import sys

import lmdb
import numpy as np
import torch
from unicore import checkpoint_utils, tasks, options, utils

sys.path.insert(0, "../DrugCLIP")

CHECKPOINT = "../edc_finetune_data/save_dir_v2/checkpoint_best.pt"
COMPLEXES = "complexes.lmdb"
PROBE_LMDB = "probe_data.lmdb"


def write_drugclip_lmdb():
    """Complexes in the key layout the task's train split expects."""
    # PDB writes CL/BR/SE/SN uppercase; dict_mol.txt has Cl/Br/Se/Sn, and a
    # mismatch silently becomes [UNK]
    FIX = {"CL": "Cl", "BR": "Br", "SE": "Se", "SN": "Sn", "SI": "Si",
           "NA": "Na", "ZN": "Zn", "FE": "Fe", "CA": "Ca", "MG": "Mg"}
    src = lmdb.open(COMPLEXES, readonly=True, lock=False, subdir=False)
    dst = lmdb.open(PROBE_LMDB, subdir=False, map_size=4 * 1024 ** 3)
    # keyed by the identifier carried in "smi", not by position: LMDB cursor
    # order is lexicographic ("0","1","10",...) while the dataset indexes
    # numerically, so pairing by counter scrambles the targets
    meta = {}
    with src.begin() as rt, dst.begin(write=True) as wt:
        i = 0
        for _, v in rt.cursor():
            r = pickle.loads(v)
            key = f"{r['receptor']}_{r['pdb_id']}"
            wt.put(str(i).encode(), pickle.dumps({
                "atoms": [FIX.get(a, a.capitalize() if len(a) > 1 else a)
                          for a in r["ligand_atoms"]],
                # the pipeline samples one conformer from a list
                "coordinates": [np.asarray(r["ligand_coordinates"], dtype=np.float32)],
                "pocket_atoms": list(r["pocket_atoms"]),
                "pocket_coordinates": np.asarray(r["pocket_coordinates"], dtype=np.float32),
                "smi": key,
                "pocket": r["receptor"],
                "label": 1,
            }))
            meta[key] = np.asarray(r["atom_min_dist"])
            i += 1
    src.close()
    dst.close()
    return meta


def main():
    meta = write_drugclip_lmdb()
    print(f"{len(meta)}개 복합체 -> {PROBE_LMDB}")

    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, input_args=[
        ".", "--user-dir", "../DrugCLIP/unimol", "--task", "drugclip",
        "--loss", "in_batch_softmax", "--arch", "drugclip", "--path", CHECKPOINT,
        "--valid-subset", "probe_data", "--batch-size", "8", "--cpu",
        "--max-pocket-atoms", "512", "--num-workers", "0", "--seed", "1",
    ])
    task = tasks.setup_task(args)
    model = task.build_model(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(CHECKPOINT)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    task.load_dataset("probe_data")
    ds = task.dataset("probe_data")
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=ds.collater)

    X, Y, E = [], [], []
    skipped = 0
    with torch.no_grad():
        for sample in loader:
            ni = sample["net_input"]
            st, dist, et = (ni["mol_src_tokens"], ni["mol_src_distance"],
                            ni["mol_src_edge_type"])
            mx = model.mol_model.embed_tokens(st)
            bias = model.mol_model.gbf_proj(model.mol_model.gbf(dist, et))
            bias = bias.permute(0, 3, 1, 2).contiguous().view(-1, dist.size(-1), dist.size(-1))
            mol_rep = model.mol_model.encoder(
                mx, padding_mask=st.eq(model.mol_model.padding_idx), attn_mask=bias)[0]

            pst, pdist, pet = (ni["pocket_src_tokens"], ni["pocket_src_distance"],
                               ni["pocket_src_edge_type"])
            px = model.pocket_model.embed_tokens(pst)
            pbias = model.pocket_model.gbf_proj(model.pocket_model.gbf(pdist, pet))
            pbias = pbias.permute(0, 3, 1, 2).contiguous().view(-1, pdist.size(-1), pdist.size(-1))
            poc_rep = model.pocket_model.encoder(
                px, padding_mask=pst.eq(model.pocket_model.padding_idx), attn_mask=pbias)[0]
            poc_pooled = poc_rep[:, 0, :]        # pocket CLS

            for b in range(st.size(0)):
                target = meta[sample["smi_name"][b]]
                n_tok = int((~st[b].eq(model.mol_model.padding_idx)).sum())
                n_atom = n_tok - 2               # CLS ... EOS
                if n_atom != len(target):
                    skipped += 1
                    continue
                atom_emb = mol_rep[b, 1:1 + n_atom, :].numpy()
                X.append(np.concatenate(
                    [atom_emb, np.tile(poc_pooled[b].numpy(), (n_atom, 1))], axis=1))
                Y.append(target)
                E.append(np.array([int(t) for t in st[b, 1:1 + n_atom]]))

    X, Y, E = np.concatenate(X), np.concatenate(Y), np.concatenate(E)
    print(f"정렬 성공 원자 {len(Y)}개 / 특징 차원 {X.shape[1]}  (복합체 {skipped}개 탈락)")

    # split by complex is not available after concatenation, so split by atom
    # with a fixed seed; the question here is representational content, not
    # generalisation across structures
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    cut = int(0.8 * len(Y))
    tr, te = perm[:cut], perm[cut:]

    def fit_report(name, feats):
        A = np.concatenate([feats, np.ones((len(feats), 1))], axis=1)
        w, *_ = np.linalg.lstsq(A[tr], Y[tr], rcond=None)
        pred = A[te] @ w
        rmse = float(np.sqrt(((pred - Y[te]) ** 2).mean()))
        mae = float(np.abs(pred - Y[te]).mean())
        print(f"  {name:<28}RMSE {rmse:.3f}   MAE {mae:.3f}")
        return rmse

    print("\n원자-포켓 최단거리 예측 (선형 프로브)")
    print(f"  {'평균값 예측 (기준선)':<28}RMSE {float(np.sqrt(((Y[tr].mean()-Y[te])**2).mean())):.3f}"
          f"   MAE {float(np.abs(Y[tr].mean()-Y[te]).mean()):.3f}")
    onehot = np.eye(int(E.max()) + 1)[E]
    fit_report("원소 종류만", onehot)
    fit_report("리간드 원자 임베딩만", X[:, :512])
    fit_report("원자 임베딩 + 포켓", X)
    print("\n참고 — BindCLIP Table 4: DrugCLIP 1.05 / BindCLIP 0.87")


if __name__ == "__main__":
    main()
