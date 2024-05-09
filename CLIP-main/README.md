## Taiyi-CLIP-Roberta Intelligent Pre-Labeling Algorithm for Image Categorization

### **Introduction of the algorithm**

Uses chinese-roberta-wwm as a Chinese language encoder, and enhances linguistic features in Chinese text processing using WWM full-word masking. Apply VIT-B-32 from CLIP model to visual coder. Pre-training is performed on Noah-Wukong (100M), Zero graphic dataset. Pre-labeled labels can be chosen to constrain 21841 class labels from Imagenet dataset as well as common 1000 class labels.

### **Innovation Points:**

1、Supporting multimodal learning, it can establish a powerful semantic relationship between image and text, and understand the image content more comprehensively.

2. Adopts RoBERTa Chinese encoder, which is more suitable for the intelligent pre-labeling task of Chinese label input.

3、Support zero sample learning, can accurately infer unknown categories, not limited to the classification category, with strong scalability.

4, can choose Imagenet's 21841 category labels and common 1000 category labels auto-filling, no need to manually give the label, to achieve one-key intelligent pre-labeling.

### **Code Runs**

Run main.py directly

### **Code Description**

main.py --run the code	

taiyi_http.py - the model to run

class_name -- store class name file	

clip -- clip model file

taiyiclip -- taiyi-clip model file

clip32--VIT-patch32's image encoder model file.

requirements.txt--dependency file

	

	
