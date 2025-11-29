torch
torchvision
Pillow
numpy
tensorboard
hdbscan
opencv_python
screeninfo
grad_cam
scikit_learn
captum==0.7.0
matplotlib==3.7.5
pykalman==0.9.7

Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/


python test__PatchGAN_MultiTasks.py --data datas/test.json --build_classifier classes_files.json --config_path Model_weight/best_hyperparams_fold_0.json --model_path Model_weight/best_model_fold_0.pth  --mode camera