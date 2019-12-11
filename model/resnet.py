from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Conv2D,Input,Add

def res_block(x, filters):
    y = Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    y = Conv2D(filters, (3, 3), padding="same", activation="relu")(y)
    y = Conv2D(filters, (1, 1), padding="same")(y)
    y = Add()([y, x])
    return y

def build_model(filters, block):

    image = Input((None,None,3))

    y = Conv2D(filters,(1,1), padding="same")(image)
    for _ in range(block):
      y = res_block(y,filters)

    y = Conv2D(3, (1, 1), padding="same")(y)
    y = Add()([y, image])
    return Model(image, y)