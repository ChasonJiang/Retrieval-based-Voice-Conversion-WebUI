import ctypes
from io import BytesIO
import os
import time
import numpy as np
import torch    
import torch.nn as nn
from infer.lib.infer_pack.attentions import MultiHeadAttention
from infer.lib.infer_pack.models import GeneratorNSF
from infer.lib import jit
from infer.lib.jit import synthesizer_jit_export,load_inputs
from infer.lib.jit.get_synthesizer import get_synthesizer

os.environ["PYTORCH_NVFUSER_DISABLE"]="fallback"


class Test(nn.Module):
    def func1(self,):
        print("func1")

    def func2(self,):
        self.func1()
        print("func2")
    
    def forward(self,x):
        return self.func2()
    
def test():
    dll_path="C:\\Users\\14404\\source\\repos\\print_tensor\\x64\\Debug\\print_tensor.dll"
    cpp_lib=ctypes.cdll.LoadLibrary(dll_path)
    # cpp_lib =ctypes.CDLL(dll_path)
    # 创建一个示例的 torch.Tensor
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sizes = data.size()
    # data_ptr = data.data_ptr()
    torch.float
    # 调用 C++ 函数
    # ctypes.POINTER(ctypes.c_float)
    data_ptr = ctypes.c_void_p(data.data_ptr())
    sizes_array = (ctypes.c_int * len(sizes))(*sizes)
    # data_ptr.ctypes
    cpp_lib.process_tensor(data_ptr, sizes_array, len(sizes))





if __name__ == "__main__":
    # test()
    device = torch.device("cuda:0")
    path="assets\\weights\\kikiV1.pth"
    model,_=get_synthesizer(path,device)
    model.eval()
    model.forward = model.infer
    # model=model.cpu()
    # inputs = load_inputs("assets/Synthesizer_inputs.pth",device=device)
    # i=[inputs["phone"],inputs["phone_lengths"],
    #          inputs["pitch"],inputs["nsff0"],
    #          inputs["sid"],inputs["rate"]]
    # model_jit=torch.jit.track(model)
    # torch.jit.save(model_jit,"assets\\weights\\kikiV1.jit")
    # model_jit
    # # print(model_jit)
    # ckpt=synthesizer_jit_export(model_path=path,mode="trace",inputs_path="assets/Synthesizer_inputs.pth",save_path="assets\\weights\\kikiV1.jit",device=device)
    # model_jit=torch.jit.load(BytesIO(ckpt["model"]),device)
    # jit.benchmark(model_jit,"assets/Synthesizer_inputs.pth",torch.device("cuda:0"),4000)


    inputs = load_inputs("assets/Synthesizer_inputs.pth",device=device)
    ckpt=synthesizer_jit_export(model_path=path,mode="script",inputs_path="assets/Synthesizer_inputs.pth",save_path="assets\\weights\\kikiV1.jit",device=device)
    model_jit=torch.jit.load(BytesIO(ckpt["model"]),device)
    # print(model_jit)
    # inputs = load_inputs("assets/Synthesizer_inputs.pth",device=device)
    # print(model_jit(**inputs))
    # inputs = load_inputs("inputs.pth",device=device)
    # print(model_jit(**inputs))

    # print("Fusion Debugging")
    # niter=10
    # with torch.jit.fuser("fuser2"):
    #     for i in range(niter):
    #         o = model_jit(**inputs)

    ####### Half floating point cannot run on the CPU, but cuda can. Everything else is fine
    jit.benchmark(model_jit,"assets/Synthesizer_inputs.pth",torch.device("cuda"),4000,is_half=False)
    # jit.benchmark(model,"assets/Synthesizer_inputs.pth",torch.device("cuda:0"),4000)


