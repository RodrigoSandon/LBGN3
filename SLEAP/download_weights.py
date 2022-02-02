import os

cmd = """

python -c "from tensorflow.keras import applications; applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

python -c "from tensorflow.keras import applications; applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
python -c "from tensorflow.keras import applications; applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
python -c "from tensorflow.keras import applications; applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.25)"
python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"

python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.35)"
python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"
python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.3)"
python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.4)"

python -c "from tensorflow.keras import applications; applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))"
python -c "from tensorflow.keras import applications; applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))"

"""


os.system(cmd)