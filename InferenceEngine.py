import ctypes
import math
import os
import threading
import time
import numpy as np
import torch
from tqdm import tqdm
from infer.lib import jit

from infer.lib.jit import load_inputs
# os.environ["PYTORCH_NVFUSER_DISABLE"]="fallback"


class C_InitResults(ctypes.Structure):
    _fields_=[
        ("state_code", ctypes.c_int),
        # ("upsample_rates", ctypes.c_float),
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

class C_LongTS(ctypes.Structure):
    _fields_=[
        ("data_ptr", ctypes.POINTER(ctypes.c_long)),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("num_dims", ctypes.c_size_t),
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
class LongTS:
    def __init__(self,tensor:torch.Tensor) -> None:
        self.value=None
        self.value=C_LongTS(
                        data_ptr=ctypes.cast(tensor.data_ptr(),ctypes.POINTER(ctypes.c_long)),
                        shape=ctypes.cast(ctypes.pointer((ctypes.c_int64 * len(tensor.size()))(*tensor.size())),ctypes.POINTER(ctypes.c_int64)),
                        num_dims=len(tensor.size())
                        )
        
class C_Inputs1(ctypes.Structure):
    _fields_=[
        ("feats", C_FloatTS),
        ("p_len", C_LongTS),
        ("cache_pitch", C_LongTS),
        ("cache_pitchf", C_FloatTS),
        ("sid", C_LongTS),
        ("rate", C_FloatTS)
    ]
    # _pack_=1

class C_Inputs2(ctypes.Structure):
    _fields_=[
        ("feats", C_FloatTS),
        ("p_len", C_LongTS),
        ("sid", C_LongTS),
        ("rate", C_FloatTS)
    ]
    # _pack_=1

class Inputs1:
    def __init__(self,feats:torch.Tensor,p_len:torch.LongTensor,
                 cache_pitch:torch.LongTensor,cache_pitchf:torch.Tensor,
                 sid:torch.LongTensor,rate:torch.FloatTensor):
        self.value=None
        c_feats=FloatTS(feats).value
        c_p_len=LongTS(p_len).value
        c_cache_pitch=LongTS(cache_pitch).value
        c_cache_pitchf=FloatTS(cache_pitchf).value
        c_sid=LongTS(sid).value
        c_rate=FloatTS(rate).value
        self.value = C_Inputs1(feats=c_feats,p_len=c_p_len,
                                cache_pitch=c_cache_pitch,
                                cache_pitchf=c_cache_pitchf,
                                sid=c_sid,rate=c_rate
                                )

class Inputs2:
    def __init__(self,feats:torch.Tensor,p_len:torch.LongTensor,
                 sid:torch.LongTensor,rate:torch.FloatTensor):
        self.value=None
        c_feats=FloatTS(feats).value
        c_p_len=LongTS(p_len).value
        c_sid=LongTS(sid).value
        c_rate=FloatTS(rate).value
        self.value = C_Inputs2(feats=c_feats,p_len=c_p_len,
                                sid=c_sid,rate=c_rate
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
        # t=threading.current_thread()
        # print(f"Current thread id: {t.ident}")
        self.engine = ctypes.cdll.LoadLibrary(dll_path)
        self.upsample_rates=0.0
        self.model_version=None
        self.model_sr=None
        self.model_f0=None
        self.model_is_half=False
        # self.infer_sem=threading.Semaphore(1)
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
        # breakpoint()
        print(f"Init Model...")
        # t=threading.current_thread()
        # print(f"Current thread id: {t.ident}")
        ckpt:dict=jit.load(model_path)
        self.model_version =ckpt.get("version","v1")
        self.model_sr =ckpt.get("config")[-1]
        self.model_f0 =ckpt.get("f0",1)
        self.model_is_half =ckpt.get("is_half",False)
        self.upsample_rates = UPSAMPE_REAETS[self.model_version][self.model_sr]
        model_byte:bytes = ckpt.get("model")

        model_device_index = self.get_device_index(ckpt.get("device"))
        # if model_device_index !=device_index:
        #     raise RuntimeError("The model inference device is inconsistent with the current inference device!")

        bytes_model = ctypes.pointer(C_BytesModel(model_byte,len(model_byte)))
        res = ctypes.pointer(C_InitResults(-1))
        self.engine.init_model(bytes_model,device_index,res)
        if res.contents.state_code!=0:
            raise RuntimeError("Failed to initialize model!")
        # self.upsample_rates = res.upsample_rates
        
        
    def infer(self,*args):
        # 通过ctypes，python与c++程序交互时，出现异常和数据不同步的问题：
        #     1. 当python传递cuda tensor时，c++就只能以cuda tensor进行操作，且修改过后的内容无法同步回来！
        #     2. 当python传递cpu tensor时，c++可以以cuda/cpu tensor进行操作，但只有以cpu tensor操作时的内容能同步回来！
        # self.infer_sem.acquire()
        # t=threading.current_thread()
        # print(f"Current thread id: {t.ident}")
        outputs_tensor = self.get_output_tensor(args[0],args[-1])
        outputs = ctypes.pointer(FloatTS(outputs_tensor).value)
        if len(args)<=4:
            inputs = ctypes.pointer(Inputs2(*args).value)
            self.engine.infer2(inputs,outputs)
        else:
            inputs = ctypes.pointer(Inputs1(*args).value)
            self.engine.infer1(inputs,outputs)
        # time.sleep(0.5)
        # self.infer_sem.release()
        return [outputs_tensor.clone().to(self.device)]


    def get_output_shape(self,feats:torch.Tensor,rate:torch.Tensor):
        dim2=math.prod(self.upsample_rates)*(feats.shape[1]-int(feats.shape[1] * (1.0-rate.item())))
        return (1,1,dim2)
    
    def get_output_tensor(self,feats:torch.Tensor,rate:torch.Tensor):
        shape = self.get_output_shape(feats,rate)
        return torch.zeros(shape,dtype=feats.dtype)

    def test(self,feats:torch.Tensor,p_len:torch.Tensor,
             cache_pitch:torch.Tensor,cache_pitchf:torch.Tensor,
             sid:torch.Tensor,rate:torch.Tensor,
             ):
        outputs_tensor = self.get_output_tensor(feats,rate)
        outputs = ctypes.pointer(FloatTS(outputs_tensor).value)
        inputs = ctypes.pointer(Inputs1(feats, p_len, cache_pitch, 
                                        cache_pitchf, sid, rate).value)
        self.engine.test(inputs,outputs)
        # print(outputs_tensor)
        return [outputs_tensor.to(self.device)]
    
    def test2(self,*args):
   
        outputs_tensor = self.get_output_tensor(args[0],args[-1]).to(self.device)
        outputs = ctypes.pointer(FloatTS(outputs_tensor).value)
        # outputs = ctypes.pointer(C_FloatTS())
        out_shape=self.get_output_shape(args[0],args[-1])
        inputs_tensor = torch.ones(out_shape,dtype=args[0].dtype).to(self.device)
        inputs = ctypes.pointer(FloatTS(inputs_tensor).value)
        # inputs = ctypes.pointer(Inputs1(*args).value)

        # outputs_tensor = self.get_output_tensor(args[0],args[-1])
        # outputs = FloatTS(outputs_tensor).value
        # out_shape=self.get_output_shape(args[0],args[-1])
        # inputs_tensor = torch.ones(out_shape,dtype=args[0].dtype)
        # inputs = FloatTS(inputs_tensor).value

        self.engine.test2(inputs,outputs)
        # data_ptr = ctypes.cast(outputs.contents.data_ptr,ctypes.POINTER(ctypes.c_float))
        # outputs_tensor = torch.from_numpy(np.ctypeslib.as_array(data_ptr,(out_shape[2])))
        # self.engine.freemem(data_ptr)
        # print(inputs_tensor.unique())
        # print(outputs_tensor.unique())
        print(outputs_tensor)
        return outputs_tensor


if __name__ =="__main__":
    # t =torch.rand([2,40000])
    # FloatTS(t)
    # t_arr=(ctypes.c_int * len(t.size()))(*t.size())

    
    ie=InferenceEngine(
        "C:\\Users\\14404\\source\\repos\\Inference Engine\\x64\\Debug\\Inference Engine.dll",
        "C:\\Users\\14404\\Project\\Retrieval-based-Voice-Conversion-WebUI\\assets\\weights\\kikiV1.jit",
        device=torch.device("cuda:0")
    )

    inputs =load_inputs("assets\Synthesizer_inputs.pth",torch.device("cuda:0"))
    # ie.test(inputs["phone"],inputs["phone_lengths"],
    #          inputs["pitch"],inputs["nsff0"],
    #          inputs["sid"],inputs["rate"])

    # inputs =load_inputs("inputs.pth",torch.device("cuda:0"))
    epoch=1000
    total_ts = 0.0
    
    # ie.infer(inputs["phone"],inputs["phone_lengths"],
    #             inputs["pitch"],inputs["nsff0"],
    #             inputs["sid"],inputs["rate"])
    # ie.infer(inputs["phone"],inputs["phone_lengths"],
    #             inputs["pitch"],inputs["nsff0"],
    #             inputs["sid"],inputs["rate"])
    # time.sleep(1)
    bar=tqdm(range(epoch))
    for i in bar:
        start_time=time.perf_counter()
        ie.infer(inputs["phone"],inputs["phone_lengths"],
                    inputs["pitch"],inputs["nsff0"],
                    inputs["sid"],inputs["rate"])
    
        total_ts+=time.perf_counter()-start_time
    print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")

    # inputs =load_inputs("inputs.pth",torch.device("cuda:0"))
    # bar=tqdm(range(epoch))
    # for i in bar:
    #     start_time=time.perf_counter()
    #     ie.infer(inputs["phone"],inputs["phone_lengths"],
    #                 inputs["pitch"],inputs["nsff0"],
    #                 inputs["sid"],inputs["rate"])
    
    #     total_ts+=time.perf_counter()-start_time
    # print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")

    # c_float_array = (ctypes.c_float * 1000)()
    # pytorch_tensor = torch.from_numpy(np.ctypeslib.as_array(c_float_array))
    # print(pytorch_tensor)
    