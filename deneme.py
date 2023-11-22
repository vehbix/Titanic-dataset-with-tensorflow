import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from data import x_train,x_test,y_train,y_test

# model.add(Dense(40,activation="sigmoid"))
# model.add(Dense(35,activation="sigmoid"))
# model.add(Dense(30,activation="sigmoid"))
# model.add(Dense(30,activation="sigmoid"))
# model.add(Dense(30,activation="relu"))
# model.add(Dense(30,activation="sigmoid"))
# model.add(Dense(25,activation="sigmoid"))
# model.add(Dense(25,activation="relu"))
# model.add(Dense(25,activation="sigmoid"))
# model.add(Dense(20,activation="sigmoid")) 
# model.add(Dense(15,activation="sigmoid")) 
# model.add(Dense(10,activation="sigmoid")) 




ep=10000
model = Sequential()
model.add(Dense(40,activation="sigmoid"))
model.add(Dense(20,activation="sigmoid"))
model.add(Dense(20,activation="sigmoid"))
model.add(Dense(20,activation="relu"))
model.add(Dense(20,activation="sigmoid"))
model.add(Dense(20,activation="sigmoid"))
model.add(Dense(20,activation="sigmoid"))  
model.add(Dense( 1,activation="relu"))
model.compile(optimizer="adam",loss="mse", metrics=['accuracy']) 
# earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=30,epochs=ep)#,callbacks=[earlyStopping]

loss_train = model.history.history['loss']
loss_val = model.history.history['val_loss']
train_acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
epochs = range(1,ep+1)
plt.plot(epochs, loss_train, 'r', label='Training loss')
plt.plot(epochs, loss_val, 'y', label='validation loss')
plt.plot(epochs, train_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='validation accuracy')
plt.title('Training Result')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# bas=150-1
# loss_train = model.history.history['loss']
# loss_val = model.history.history['val_loss']
# train_acc = model.history.history['accuracy']
# val_acc = model.history.history['val_accuracy']
# epochs = range(bas+1,ep+1)
# plt.plot(epochs, loss_train[bas:], 'r', label='Training loss')
# plt.plot(epochs, loss_val[bas:], 'y', label='validation loss')
# plt.plot(epochs, train_acc[bas:], 'b', label='Training accuracy')
# plt.plot(epochs, val_acc[bas:], 'g', label='validation accuracy')
# plt.title('Training Result')
# plt.xlabel('Epochs')
# plt.legend()
# plt.show()


# modelKaybi = pd.DataFrame(model.history.history)
# modelKaybi.plot()
# plt.show()




