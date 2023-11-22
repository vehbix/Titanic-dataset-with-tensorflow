import pandas as pd
from egitFonksiyonlar import egit, egitim, egitPlt
import tensorflow as tf


Adamax=tf.keras.optimizers.Adamax(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999)#en yüksek doğruluk
Adam=tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)#en düşük standart sapma
RMSprop=tf.keras.optimizers.RMSprop(learning_rate = 0.001, rho = 0.9)
Adadelta=tf.keras.optimizers.Adadelta(learning_rate = 1.0, rho = 0.95)


a=egitim(optimizer="adam",tekrar=3,min_acc=100,batch_size=30,max_epoch=50)
a.calis()
a.sonucYazdir()
a.sonucKaydet()

 
