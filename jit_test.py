from collections import OrderedDict
from io import BytesIO
import pickle
import torch

def get_synthesizer(pth_path, device=torch.device("cpu")):
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
        SynthesizerTrnMs256NSFsidOnlyEncode,
        DecodeForSynthesizer
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    # tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    config = cpt["config"]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    synthesizer_config = [*config[:9],*config[-3:]]
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsidOnlyEncode(*synthesizer_config)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*synthesizer_config)
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*synthesizer_config, is_half=False)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*synthesizer_config)
    del net_g.enc_q
    decoder_config = [config[2],*config[-9:-3],*config[-2:]]
    decoder = DecodeForSynthesizer(*decoder_config)
    cpt_weight:dict = cpt["weight"]
    synthesizer_weight = {
        k:v for k, v in cpt_weight.items() if "dec" not in k
    }
    decoder_weight = {
        k:v for k, v in cpt_weight.items() if "dec" in k
    }
    net_g.load_state_dict(synthesizer_weight, strict=False)
    decoder.load_state_dict(decoder_weight, strict=False)
    decoder = decoder.float()
    net_g = net_g.float()
    decoder.eval().to(device)
    net_g.eval().to(device)
    return net_g, decoder, cpt

def export(
    model: torch.nn.Module,
    device=torch.device("cpu"),
    is_half: bool = False,
) -> dict:
    model = model.half() if is_half else model.float()
    model.eval()
    model_jit = torch.jit.script(model)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    buffer = BytesIO()
    # model_jit=model_jit.cpu()
    torch.jit.save(model_jit, buffer)
    del model_jit
    model_bytes = buffer.getvalue()
    # cpt["is_half"] = is_half
    return model_bytes


def load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save(ckpt: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(ckpt, f)

def synthesizer_jit_export(
    model_path: str,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    encoder,decoder, cpt = get_synthesizer(model_path, device)
    assert isinstance(cpt, dict)
    encoder_jit = export(encoder, device, is_half)
    decoder_jit = export(decoder, device, is_half)
    cpt.pop("weight")
    cpt["model"] = {
        "encoder":encoder_jit,
        "decoder":decoder_jit
    }
    cpt["device"] = device
    cpt["is_half"] = is_half
    save(cpt, save_path)
    return cpt


if __name__ == "__main__":
    device = torch.device("cuda:0")
    pth_path = "assets/weights/kikiV1.pth"
    # encoder, decoder, cpt=get_synthesizer(pth_path,device)
    ckpt=synthesizer_jit_export(pth_path)



