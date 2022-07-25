## Install
``` python
pip install -r requirements.txt
```

## Pretrained Model  
### prepare two floders: weigth/dqt_hinet and weigth/swinir
Please downlaod two pretrained models:
```
hinet: https://drive.google.com/file/d/1bhjUDbq-TYp7GN146ghwEnk4BLM_ufV8/view?usp=sharing
swinir: https://drive.google.com/file/d/1OY8BV2A9VeIUsA2aPpxUCcpLxsxvyMVX/view?usp=sharing
```
Put hinet model:dqt_hinet_flikr2k_p512_epoch57.pth in path: weight/dqt_hinet 

Put 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth in weight/swinir

## Inference
``` python
python src/eval_qm.py --realdatapath <your_image_val_path> --model <hinet_pretrained_model_path>
python swinir/main_test_swinir.py 
``` 
The final result images will be saved in path: results/swinir_classical_sr_x4