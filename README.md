<center><h1>VegDet</h1></center>

## Install
 - Fedora
```
sudo dnf instal python3.9
python3.9 -m pip install git+https://github.com/WizzyGeek/VegDet.git
```

- Everyone Else
```
python -V
```
check if python 3.9
```
python -m pip install git+https://github.com/WizzyGeek/VegDet.git
```

## Run

```
vegclass ./PathToYourImage.png
```

and (detection model is not accurate at the moment)
```
vegdet ./PathToYourImage.png
```

Plans to extend to functionality are shaky, but the framework is present

### Aim (Why?)
To implement an affordable classifcation and detection models for edge applications

### Citations

 - Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems,
> Software available from tensorflow.org.
 - Roboflow
 - arXiv:1911.09070
   > Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. doi:10.48550/ARXIV.1905.11946
 - M. Israk Ahmed, &amp; Shahriyar Mahmud Mamun. (2021). <i>Vegetable Image Dataset</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/2965251
   > For classifier model dataset

> *Thanks to Google colab, for providing free GPU compute, which greatly helped this project!*