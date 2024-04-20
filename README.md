# Python library : torch_cmif

The torch_cmif library provides a fast implementation the cross mutual information between one real image and one another on PyTorch.
<br />
<br />
<br />
In this library, the paper from J. Öfverstedt et al. has been implemented :
<br />


# References :

- Johan Öfverstedt, Joakim Lindblad, Nataša Sladoje, (2022). Fast computation of mutual information in the frequency domain with applications to global multimodal image alignment, - https://www.sciencedirect.com/science/article/pii/S0167865522001817


<hr />


# Install library



```bash
%%bash
if !python -c "import torch_cmif" 2>/dev/null; then
    pip install https://github.com/Simon-Bertrand/FastCMIF-PyTorch/archive/main.zip
fi
```

# Import library



```python
import torch_cmif
```

    Obtaining file:///home/sbertrand/D%C3%A9veloppement/repos/gits/torch-cmif
      Installing build dependencies ... [?25ldone
    [?25h  Checking if build backend supports build_editable ... [?25ldone
    [?25h  Getting requirements to build editable ... [?25ldone
    [?25h  Installing backend dependencies ... [?25ldone
    [?25h  Preparing editable metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: torch>=2.2.1 in ./.venv/lib/python3.10/site-packages (from torch_cmif==0.0.1) (2.2.2)
    Requirement already satisfied: numpy==1.26.4 in ./.venv/lib/python3.10/site-packages (from torch_cmif==0.0.1) (1.26.4)
    Requirement already satisfied: filelock in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (3.13.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (4.11.0)
    Requirement already satisfied: sympy in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (1.12)
    Requirement already satisfied: networkx in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (3.3)
    Requirement already satisfied: jinja2 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (3.1.3)
    Requirement already satisfied: fsspec in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (2024.3.1)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.105)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.105)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.105)
    Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (8.9.2.26)
    Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.3.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (11.0.2.54)
    Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (10.3.2.106)
    Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (11.4.5.107)
    Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.0.106)
    Requirement already satisfied: nvidia-nccl-cu12==2.19.3 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (2.19.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (12.1.105)
    Requirement already satisfied: triton==2.2.0 in ./.venv/lib/python3.10/site-packages (from torch>=2.2.1->torch_cmif==0.0.1) (2.2.0)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in ./.venv/lib/python3.10/site-packages (from nvidia-cusolver-cu12==11.4.5.107->torch>=2.2.1->torch_cmif==0.0.1) (12.4.127)
    Requirement already satisfied: MarkupSafe>=2.0 in ./.venv/lib/python3.10/site-packages (from jinja2->torch>=2.2.1->torch_cmif==0.0.1) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in ./.venv/lib/python3.10/site-packages (from sympy->torch>=2.2.1->torch_cmif==0.0.1) (1.3.0)
    Building wheels for collected packages: torch_cmif
      Building editable for torch_cmif (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for torch_cmif: filename=torch_cmif-0.0.1-0.editable-py3-none-any.whl size=15226 sha256=d747e82187f8d874a980a66681477ba6d4925ca8f63799e2a272c25d89c62424
      Stored in directory: /tmp/pip-ephem-wheel-cache-lkoisnbe/wheels/d3/b2/4f/76de76a0114a7bc162a4b992efd2e64a11f11f227e1a19a6cb
    Successfully built torch_cmif
    Installing collected packages: torch_cmif
      Attempting uninstall: torch_cmif
        Found existing installation: torch_cmif 0.0.1
        Uninstalling torch_cmif-0.0.1:
          Successfully uninstalled torch_cmif-0.0.1
    Successfully installed torch_cmif-0.0.1
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
!pip install -q matplotlib torchvision
import torch
import matplotlib.pyplot as plt
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


## LOAD IMAGE AND TEST IF RANDOM EXTRACTED CENTER POSITIONS ARE CORRECTLY FOUND


Install notebook dependencies



```python
!pip install -q requests
import requests
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


Load Mandrill image



```python
import tempfile
import torchvision
import torch.nn.functional as F

with tempfile.NamedTemporaryFile() as fp:
    fp.write(
        requests.get(
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/Mandrill-k-means.png"
        ).content
    )
    im = F.interpolate(
        (
            torchvision.io.read_image(
                fp.name, torchvision.io.ImageReadMode.RGB
            )
            .unsqueeze(0)
            .to(torch.float64)
            .div(255)
        ),
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
    )
```

Run multiple tests to check if random crop center is correclty found by the ZNCC.



```python
import random

success = 0
failed = 0
pts = []
for _ in range(16):
    imH = random.randint(64, 128)
    imW = random.randint(64, 128)
    i = random.randint(imH // 2 + 1, im.size(-2) - imH // 2 - 1)
    j = random.randint(imW // 2 + 1, im.size(-1) - imW // 2 - 1)

    imT = im[
        :, :, i - imH // 2 : i + imH // 2 + 1, j - imW // 2 : j + imW // 2 + 1
    ]
    if (
        (
            torch_cmif.FastCMIF.findArgmax(
                torch_cmif.FastCMIF(8, "none")(im, imT)
            )
            - torch.Tensor([[[i]], [[j]]])
        ).abs()
        < 3
    ).all():
        pts += [
            dict(
                i=i,
                imH=imH,
                imW=imW,
                j=j,
                success=True,
            )
        ]
        success += 1
    else:
        pts += [
            dict(
                i=i,
                imH=imH,
                imW=imW,
                j=j,
                success=False,
            )
        ]
        failed += 1

plt.imshow(im[0].moveaxis(0, -1))
ax = plt.gca()
for pt in pts:
    ax.add_patch(
        plt.Rectangle(
            (pt["j"] - pt["imW"] // 2, pt["i"] - pt["imH"] // 2),
            pt["imW"],
            pt["imH"],
            edgecolor="g" if pt["success"] else "r",
            facecolor="none",
            linewidth=0.5,
        )
    )
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](figs/README_14_1.png)
    



```python
ans = torch_cmif.FastCMIF(8, "sum")(im, imT)
plt.imshow(ans[0].mean(0))
plt.title("CMIF")
```




    Text(0.5, 1.0, 'CMIF')




    
![png](figs/README_15_1.png)
    



```python
%timeit torch_cmif.FastCMIF(8, "sum")(im, imT)
```

    164 ms ± 4.58 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


Total errors :



```python
dict(success=success, failed=failed)
```




    {'success': 16, 'failed': 0}




```python

```
