# Heuristic Style Transfer for Real-Time, Efficient Weather Attribute Detection

## 📌 Overview

This project explores **style-based neural architectures** for real-time **weather classification** using advanced deep learning techniques. Our models integrate **PatchGAN, Gram Matrices, and Attention Mechanisms** to efficiently extract weather-related features from images. The pre-trained weights for the PM and PMG models (under 50 MB) are located in the Model_weight folder.
You can download the pre-trained weights for all models here: https://drive.google.com/drive/folders/1W_4oVFMMREbiC0WJcQQmnM7evl4j5fV0?usp=sharing .

## 💡 Idea behind our article



Find a better quality video here: https://youtu.be/JHXpfte628s

https://github.com/user-attachments/assets/1600d3f9-f02c-44fa-9f44-3c70312349f4







 
## 🎥 Real-time demonstration

### Raspberry Pi
**You can find the ready-to-use script to install the models on Raspberry Pi here** : https://github.com/Hamedkiri/Embedded_system_rasberry

### Demonstration with a world tour

Find a better quality video here: https://youtu.be/hWglszOLNpg?si=zYHB_EMw8qjKdjgh




https://github.com/user-attachments/assets/bcc522a2-a49a-4988-8ba2-b00aff69bd73









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
python test_PMG.py --data datas/test.json \
    --build_classifier classes_files.json \
    --config_path Model_weights/PMG/hyperparameters_PMG.json \
    --model_path Model_weights/PMG/best_model_PMG.pth \
    --mode camera
```

### ⚠️ Important Notes
- Specify the **hyperparameter configuration file** using `--config_path` to correctly reconstruct the model architecture.
- Use `--build_classifier` to define the **tasks and class mappings**.

---

## Distribution of the dataset
Find the data to test or train the model here : https://github.com/Hamedkiri/Weather_MultiTask_Datasets .
## Benchmark validation

To test the benchmarks, use the --mode benchmark option and specify the mapping file between your classes and those of the benchmark.

```bash
python test_PMG.py --build_classifier datas/classes_files.json \ 
    --benchmark_mapping Test_benchmark/Match_between_our_classes_and_benchmarks.json \
    --config_path Model_weights/PMG/hyperparameters_PMG.json \ 
    --model_path Model_weights/PMG/best_model_PMG.pth \
    --mode benchmark \
    --benchmark_folder benchmark_dataset \ 
    --save_dir results_test

```
Stay tuned for updates! 📢 We will be releasing more resources soon.

