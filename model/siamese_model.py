# siamese_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from keras_cv_attention_models import mobilevit

def build_encoder(input_shape=(224, 224, 3)):
    """
    Encoder using MobileViT backbone.
    Input: RGB images (224x224x3)
    Output: 128-dimensional embedding
    """
    inputs = layers.Input(shape=input_shape)

    # MobileViT backbone (small) for feature extraction
    base_model = mobilevit.MobileViT_S(
        input_shape=input_shape,
        pretrained=True,
        num_classes=0  # disables top classification layer
    )
    x = base_model(inputs)

    # Global average pooling + embedding projection
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs, x, name="Encoder")


def build_siamese_network(input_shape=(224, 224, 3)):
    """
    Build a Siamese network using the MobileViT encoder.
    """
    encoder = build_encoder(input_shape)

    # Two inputs for the Siamese network
    input_1 = layers.Input(shape=input_shape)
    input_2 = layers.Input(shape=input_shape)

    # Pass through shared encoder
    emb_1 = encoder(input_1)
    emb_2 = encoder(input_2)

    # L1 distance
    distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb_1, emb_2])
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model([input_1, input_2], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
