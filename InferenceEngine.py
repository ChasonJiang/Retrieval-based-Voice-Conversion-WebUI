import ctypes
from io import BytesIO
import math
import os
import threading
import time
import numpy as np
import torch
from tqdm import tqdm
from infer.lib import jit
# from tools import jit

from infer.lib.jit import load_inputs
# os.environ["PYTORCH_NVFUSER_DISABLE"]="fallback"


class C_InitResults(ctypes.Structure):
    _fields_=[
        ("state_code", ctypes.c_int),
    ]
    # _pack_=1

class C_BytesModel(ctypes.Structure):
    _fields_=[
        ("data_ptr", ctypes.c_char_p),
        ("size", ctypes.c_size_t),
    ]
    # _pack_=1

class C_FloatTS(ctypes.Structure):
    _fields_=[
        ("data_ptr", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("num_dims", ctypes.c_size_t),
        ("is_half", ctypes.c_bool)
    ]
    # _pack_=1


class FloatTS:
    def __init__(self,tensor:torch.Tensor) -> None:
        self.value=None
        is_half=False  
        if tensor.dtype == torch.half:
            is_half=True
        self.value=C_FloatTS(
                        data_ptr=ctypes.cast(tensor.data_ptr(),ctypes.POINTER(ctypes.c_float)),
                        shape=ctypes.cast(ctypes.pointer((ctypes.c_int64 * len(tensor.size()))(*tensor.size())),ctypes.POINTER(ctypes.c_int64)),
                        num_dims=len(tensor.size()),
                        is_half=is_half
                        )

        
class C_Inputs1(ctypes.Structure):
    _fields_=[
        ("z_masked", C_FloatTS),
        ("nsff0", C_FloatTS),
        ("g", C_FloatTS),
    ]
    # _pack_=1

class C_Inputs2(ctypes.Structure):
    _fields_=[
        ("z_masked", C_FloatTS),
        ("g", C_FloatTS),
    ]
    # _pack_=1


        
class Inputs1:
    def __init__(self,z_masked:torch.FloatTensor,
                 nsff0:torch.FloatTensor,
                 g:torch.FloatTensor):
        self.value=None
        c_z_masked=FloatTS(z_masked).value
        c_nsff0=FloatTS(nsff0).value
        c_g=FloatTS(g).value
        self.value = C_Inputs1(z_masked=c_z_masked,
                                nsff0=c_nsff0,
                                g=c_g
                                )
        
class Inputs2:
    def __init__(self,z_masked:torch.FloatTensor,
                 g:torch.FloatTensor):
        self.value=None
        c_z_masked=FloatTS(z_masked).value
        c_g=FloatTS(g).value
        self.value = C_Inputs1(z_masked=c_z_masked,
                                g=c_g
                                )

UPSAMPE_REAETS={
    "v1":{
        32000:[10,4,2,2,2],
        40000:[10,10,2,2],
        48000:[10,6,2,2,2]
    },
    "v2":{
        32000:[10,8,2,2],
        48000:[12,10,2,2]
    }
}


class InferenceEngine:
    def __init__(self,dll_path:str,model_path=None,device:torch.device=torch.device("cpu")) -> None:
        self.model_path = model_path
        self.device = device
        self.device_index=self.get_device_index(self.device)
        print(f"Loading Inference Engine...")
        self.engine = ctypes.cdll.LoadLibrary(dll_path)
        self.upsample_rates=0.0
        self.model_version=None
        self.model_sr=None
        self.model_f0=None
        self.model_is_half=False
        self.encoder:torch.nn.Module = None
        if model_path:
            self.init_model(self.model_path,self.device_index)



    def get_device_index(self,device):
        if "cpu" == str(device):
            device_index = -1
        elif "cuda" == str(device):
            device_index=0
        elif "cuda:" in str(device):
            device_index = int(str(device).split(":")[-1])
        else:
            raise ValueError(f"Unsupported devices:{device}")
        return device_index

    def init_model(self,model_path:str,device_index:int):
        print(f"Init Model...")
        try:
            ckpt:dict=jit.load(model_path)
            self.model_version =ckpt.get("version","v1")
            self.model_sr =ckpt.get("config")[-1]
            self.model_f0 =ckpt.get("f0",1)
            self.model_is_half =ckpt.get("is_half",False)
            self.upsample_rates = UPSAMPE_REAETS[self.model_version][self.model_sr]
            model_bytes:dict = ckpt.get("model")

            self.encoder = torch.jit.load(BytesIO(model_bytes["encoder"]),map_location=self.device)
            self.encoder = self.encoder.to(self.device)

            bytes_model = ctypes.pointer(C_BytesModel(model_bytes["decoder"],len(model_bytes["decoder"])))
            res = ctypes.pointer(C_InitResults(-1))
            self.engine.init_model(bytes_model,device_index,res)
            if res.contents.state_code!=0:
                raise RuntimeError("Failed to initialize model!")
        except Exception as e:
            print(e)



    def get_output_shape(self,feats:torch.Tensor,rate:torch.Tensor):
        dim2=math.prod(self.upsample_rates)*(feats.shape[1]-int(feats.shape[1] * (1.0-rate.item())))
        return (1,1,dim2)
    
    def get_output_tensor(self,feats:torch.Tensor,rate:torch.Tensor):
        shape = self.get_output_shape(feats,rate)
        return torch.zeros(shape,dtype=feats.dtype)
        
    def infer(self,*args):
        # 通过ctypes，python与c++程序交互时，出现异常和数据不同步的问题：
        #     1. 当python传递cuda tensor时，c++就只能以cuda tensor进行操作，且修改过后的内容无法同步回来！
        #     2. 当python传递cpu tensor时，c++可以以cuda/cpu tensor进行操作，但只有以cpu tensor操作时的内容能同步回来！
        
        outputs_tensor = self.get_output_tensor(args[0],args[-1])
        outputs = ctypes.pointer(FloatTS(outputs_tensor).value)
        if len(args)<=4:
            z_masked, g= self.encoder(*args)
            inputs = ctypes.pointer(Inputs2(z_masked, g).value)
            self.engine.infer2(inputs,outputs)
        else:
            z_masked,nsff0,g= self.encoder(*args)
            inputs = ctypes.pointer(Inputs1(z_masked, nsff0, g).value)
            self.engine.infer1(inputs,outputs)
        # print(outputs_tensor)
        return [outputs_tensor.clone().to(self.device)]
    


if __name__ =="__main__":

    ie=InferenceEngine(
        "C:\\Users\\14404\\source\\repos\\Inference Engine 2\\x64\\Debug\\Inference Engine.dll",
        "C:\\Users\\14404\\Project\\Retrieval-based-Voice-Conversion-WebUI\\assets\\weights\\kikiV1.jit",
        device=torch.device("cuda:0")
    )

    inputs =load_inputs("assets\Synthesizer_inputs.pth",torch.device("cuda:0"))

    # ie.infer(inputs["phone"],inputs["phone_lengths"],
    #             inputs["pitch"],inputs["nsff0"],
    #             inputs["sid"],inputs["rate"])
    # ie.infer(inputs["phone"],inputs["phone_lengths"],
    #             inputs["pitch"],inputs["nsff0"],
    #             inputs["sid"],inputs["rate"])
    

    # time.sleep(1)
    epoch=1000
    total_ts = 0.0
    bar=tqdm(range(epoch))
    for i in bar:
        start_time=time.perf_counter()
        ie.infer(inputs["phone"],inputs["phone_lengths"],
                    inputs["pitch"],inputs["nsff0"],
                    inputs["sid"],inputs["rate"])
    
        total_ts+=time.perf_counter()-start_time
    print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")

    