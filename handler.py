import logging
import os
import re
import traceback

import esm
import runpod
import torch

from interprot.esm_wrapper import ESM2Model
from interprot.sae_model import SparseAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = "weights"
SAE_NAME_TO_CHECKPOINT = {
    "SAE4096-L24": "esm2_plm1280_l24_sae4096_100Kseqs.pt",
}


def load_models():
    sea_name_to_info = {}
    for sae_name, sae_checkpoint in SAE_NAME_TO_CHECKPOINT.items():
        pattern = r"plm(\d+).*?l(\d+).*?sae(\d+)"
        matches = re.search(pattern, sae_checkpoint)
        if matches:
            plm_dim, plm_layer, sae_dim = map(int, matches.groups())
        else:
            raise ValueError("Checkpoint file must start with plm<n>_l<n>_sae<n>")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ESM2 model
        logger.info(f"Loading ESM2 model with plm_dim={plm_dim}")
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        esm2_model = ESM2Model(
            num_layers=33,
            embed_dim=plm_dim,
            attention_heads=20,
            alphabet=alphabet,
            token_dropout=False,
        )
        esm2_weights = os.path.join(WEIGHTS_DIR, "esm2_t33_650M_UR50D.pt")
        esm2_model.load_esm_ckpt(esm2_weights)
        esm2_model = esm2_model.to(device)

        # Load SAE models (ensure compatibility with Lightning checkpoints)
        logger.info(f"Loading SAE model {sae_name}")
        sae_model = SparseAutoencoder(plm_dim, sae_dim).to(device)
        sae_weights = os.path.join(WEIGHTS_DIR, sae_checkpoint)
        state_dict = torch.load(sae_weights, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {k.replace("sae_model.", ""): v for k, v in state_dict.items()}
        sae_model.load_state_dict(state_dict)

        sea_name_to_info[sae_name] = {
            "model": sae_model,
            "plm_layer": plm_layer,
        }

    logger.info("Models loaded successfully")
    return esm2_model, sea_name_to_info


def handler(event):
    logger.info(f"starting handler with event: {event}")
    try:
        input_data = event["input"]
        seq = input_data["sequence"]
        sae_name = input_data["sae_name"]
        dim = input_data.get("dim")
        sae_info = sea_name_to_info[sae_name]
        sae_model = sae_info["model"]
        plm_layer = sae_info["plm_layer"]
        logger.info(f"sae_name: {sae_name}, plm_layer: {plm_layer}, dim: {dim}")

        _, esm_layer_acts = esm2_model.get_layer_activations(seq, plm_layer)
        esm_layer_acts = esm_layer_acts[0].float()

        sae_acts = sae_model.get_acts(esm_layer_acts)[1:-1]

        data = {}
        if dim is not None:
            sae_dim_acts = sae_acts[:, dim].cpu().numpy()
            data["tokens_acts_list"] = [round(float(act), 1) for act in sae_dim_acts]
        else:
            max_acts, _ = torch.max(sae_acts, dim=0)
            sorted_dims = torch.argsort(max_acts, descending=True)
            active_dims = sorted_dims[max_acts[sorted_dims] > 0]
            sae_acts_by_active_dim = sae_acts[:, active_dims].cpu().numpy()

            data["token_acts_list_by_active_dim"] = [
                {
                    "dim": int(active_dims[dim_idx].item()),
                    "sae_acts": [
                        round(float(act), 1) for act in sae_acts_by_active_dim[:, dim_idx]
                    ],
                }
                for dim_idx in range(sae_acts_by_active_dim.shape[1])
            ]

        return {
            "status": "success",
            "data": data,
        }
    except Exception as e:
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


esm2_model, sea_name_to_info = load_models()
runpod.serverless.start({"handler": handler})
