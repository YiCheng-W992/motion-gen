## 环境配置

```
conda create -n motion python=3.10 -y
conda activate motion
```

```
pip install -r requirements.txt
```

```
mkdir language_models
hf download distilbert/distilbert-base-uncased --local-dir ./language_models/distilbert-base-uncased
```

模型放置于save 目录下

## 目录结构

```
minimal_trans_dec_infer/
  infer.py
  model/
    mdm.py
    mdm_modules.py
    rotation2xyz.py
    BERT/
      BERT_encoder.py
  diffusion/
    gaussian_diffusion.py
    respace.py
    nn.py
    losses.py
    dpm_solver_pytorch.py
    spacing/
      __init__.py
      timesteps.py
  utils/
    model_util.py
    parser_util.py
    dist_util.py
    fixseed.py
    sampler_util.py
    misc.py
    loss_util.py
  data_loaders/
    tensors.py
    humanml_utils.py
    humanml/
      common/
        quaternion.py
      scripts/
        motion_process.py
```

## 运行推理
在本目录执行：
```
python infer.py \
  --model_path save/humanml_dec/model000600000.pt \
  --text_prompt "a person walks forward" \
  --num_samples 1 \
  --num_repetitions 1 \
  --motion_length 6 \
  --sample_steps 50 \
  --out outputs/sample.npy
```

输出 `.npy` 是一个 dict：
- `motion`：`[N, J, 3, T]` 的 XYZ 关节（HumanML，22 joints）
- `text`：文本列表
- `lengths`：序列长度
- `fps`：20

## 启动服务（FastAPI）
在本目录执行：
```
bash serve/run_server.sh ../save/humanml_trans_dec_512_bert/model000200000.pt --port 8000 --device 0
```

请求示例：
```
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"text":"a person walks forward"}' --output results.npy
```

