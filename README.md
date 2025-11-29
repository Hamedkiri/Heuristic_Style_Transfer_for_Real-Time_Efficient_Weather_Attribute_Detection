# Heuristic Style Transfer for Real-Time, Efficient Weather Attribute Detection

## 📌 Overview

This project explores **style-based neural architectures** for real-time **weather classification** using advanced deep learning techniques. Our models integrate **PatchGAN, Gram Matrices, and Attention Mechanisms** to efficiently extract weather-related features from images.

## 💡 Idea behind our article

https://github.com/user-attachments/assets/1600d3f9-f02c-44fa-9f44-3c70312349f4





Find a better quality video here: https://youtu.be/JHXpfte628s



 
## 🎥 Real-time demonstration












https://github.com/user-attachments/assets/bcc522a2-a49a-4988-8ba2-b00aff69bd73





Find a better quality video here: https://youtu.be/hWglszOLNpg?si=zYHB_EMw8qjKdjgh




## 🚀 Package Installation

Before running the model, ensure you have **Python 3.8.19** installed. You can create a virtual environment and install the necessary dependencies with:

```bash
pip install -r requirements.txt
```
See the "readme_installation_on_window.txt" file for installation on Windows operating system

## 🎯 Quick Test: PatchGAN-MultiTasks (PM) Model

Our **PatchGAN-MultiTasks (PM)** model, with only **3,188,736 parameters**, is optimized for real-time execution. We will soon publish the **dataset** along with **detailed explanations** on how to perform various tests, including:

- **Grad-CAM** & **T-SNE** visualizations 🖼️
- **Modularity tests** by selectively removing tasks 🔍
- Performance validation against our published results 📊

### ✅ Real-Time Inference with a Camera

To test the model in real time using your **camera**, execute the following command:

```bash
python test_PM.py --data datas/test.json \
    --build_classifier classes_files.json \
    --config_path Model_weight/best_hyperparams_fold_0.json \
    --model_path Model_weight/best_model_fold_0.pth \
    --mode camera
```

### ⚠️ Important Notes
- Specify the **hyperparameter configuration file** using `--config_path` to correctly reconstruct the model architecture.
- Use `--build_classifier` to define the **tasks and class mappings**.

---

Stay tuned for updates! 📢 We will be releasing more resources and datasets soon.
## Distribution of the dataset
![Dataset_repartition.png](images/Dataset_repartition.png)
## Benchmark validation

To test the benchmarks, use the --mode benchmark option and specify the mapping file between your classes and those of the benchmark.

```bash
python test_PM.py --build_classifier datas/classes_files.json \ 
    --benchmark_mapping Test_benchmark/classes_correspondance_benchmark.json  \
    --config_path Model_weight/PatchGAN_MultiTasks/best_model_fold_0_hyperparameters.json \ 
    --model_path Model_weight/PatchGAN_MultiTasks/best_model_fold_0.pth \
    --mode benchmark \
    --benchmark_folder benchmark_dataset \ 
    --save_dir results_test

```

