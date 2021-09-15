# Development of Deep Neural Network for the Classification of the Aortic Valve using Echocardiographic Data
This repository hosts the code implemented for the thesis entitled "Development of Deep Neural Network for the Classification of the Aortic Valve using Echocardiographic Data" by Stylianos Zafeiris, School of Electrical and Computer Engineering, Technical University of Crete ([TUC](https://www.tuc.gr)).

**Thesis comitee:**
* Michalis Zervakis (Supervisor)
* George Chalkiadakis
* Konstantinos Marias ([FORTH](https://www.forth.gr/index_main.php?l=e))

## Abstract
The heart is one of the most important organs of the human body, which is responsible for the circulation of blood in it. Many times, however, various cardiovascular diseases cause problems in its functionality and need immediate treatment. These diseases are either caused by lifestyle, or exist in the form of congenital anomalies and cause problems later in the patient’s life. One such abnormality is the bicuspid aortic valve, which affects approximately 1% to 2% of the world's population. It might cause various other cardiovascular diseases such as aortic valve stenosis, which can cause decreased blood flow to the aorta, which is the main artery of the human body. Hence, a fast and accurate diagnosis of the aortic valve type is important for the immediate treatment of possible diseases. The most immediate way to detect the type of aortic valve is by an echocardiogram. In some occasions, the noisy nature of ultrasound makes it difficult for doctors to diagnose.
This study aims to distinguish the aortic valve into bicuspid (abnormal) and tricuspid (normal), from echocardiographic data, in order to facilitate specialists during the examination of patients. Aortic valve classification is achieved using deep convolutional neural networks and specifically the well-known 2D network, VGG16, which is extended to 3D. Various techniques, such as data augmentation and transfer learning, are used to address the limitation of the small amount of available data. The proposed architecture achieves an accuracy of 93.82% up to 98.64%, which makes it capable of being used to assist cardiologists during the diagnosis.

