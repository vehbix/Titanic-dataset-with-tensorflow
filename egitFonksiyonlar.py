import pandas as pd
import tensorflow as tf
import statistics
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from data import x_train,x_test,y_train,y_test
from matematikselFonksiyonlar import standartSapmaBul





def egitPlt(optimizer,batch,max_epoch):
    model = Sequential()
    model.add(Dense(40,activation="sigmoid"))
    model.add(Dense(20,activation="sigmoid"))
    model.add(Dense(20,activation="sigmoid"))
    model.add(Dense(20,activation="relu"))
    model.add(Dense(20,activation="sigmoid"))
    model.add(Dense(20,activation="sigmoid"))
    model.add(Dense(20,activation="sigmoid"))  
    model.add(Dense( 1,activation="relu"))
    model.compile(optimizer=optimizer,loss="mse", metrics=['accuracy']) 
    earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=100)
    batch=int(len(y_train)/batch)+1
    model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),
              batch_size=batch,epochs=max_epoch,callbacks=[earlyStopping])
    modelKaybi = pd.DataFrame(model.history.history)
    modelKaybi.plot()
    plt.show()
    return round(model.history.history['accuracy'][-1]*100,2)


class egit():
    def __init__(self,optimizer,batch_size,max_epoch):
        self.optimizer=optimizer        
        self.max_epoch=max_epoch
        self.batch_size=batch_size
    
    def run(self):
        model = Sequential()
        model.add(Dense(40,activation="sigmoid"))
        model.add(Dense(20,activation="sigmoid"))
        model.add(Dense(20,activation="sigmoid"))
        model.add(Dense(20,activation="relu"))
        model.add(Dense(20,activation="sigmoid"))
        model.add(Dense(20,activation="sigmoid"))
        model.add(Dense(20,activation="sigmoid"))  
        model.add(Dense( 1,activation="relu"))
        model.compile(optimizer=self.optimizer,loss="mse", metrics=['accuracy']) 
        earlyStopping=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=100)
        model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),
                batch_size=self.batch_size,epochs=self.max_epoch,callbacks=[earlyStopping])
        return round(model.history.history['accuracy'][-1]*100,2)
    

class egitim():
    def __init__(self,optimizer,tekrar,min_acc,batch_size,max_epoch):
        self.optimizer=optimizer
        self.tekrar=tekrar
        self.min_acc=min_acc
        self.batch_size=batch_size
        self.max_epoch=max_epoch
    def calis(self):    
        bas=time.time() 
        acc_list=[egit(self.optimizer,self.batch_size,self.max_epoch).run()]
        bad_acc=[]
        sayac=1
        
        for i in range(self.tekrar-1):
            if acc_list[-1]<self.min_acc:
                acc_list.append(egit(self.optimizer,self.batch_size,self.max_epoch).run())    
                sayac+=1 
            else:
                break    
        while True:
            for i in acc_list:
                if statistics.mean(acc_list)-i>5:
                    bad_acc.append(i)
                    acc_list.remove(i) 
                    a=egit(self.optimizer,self.batch_size,self.max_epoch).run()
                    sayac+=1
                    if(statistics.mean(acc_list)-a<5):
                        acc_list.append(a)
                    else:
                        bad_acc.append(a)
            if int(len(acc_list))==self.tekrar:
                break
            else:
                acc_list.append(egit(self.optimizer,self.batch_size,self.max_epoch).run())
                sayac+=1
        son=time.time()
        self.sayac=sayac
        self.bad_acc=bad_acc        
        self.gecen=round(son-bas,1)
        self.acc_list=acc_list    
           
        self.cikti='{} {:.2f}\n{} {:.2f}\n{} {:.2f}\n{} {}\n{} {}\n{}{}\n{}{} {}\n{} {}\n{} {}\n{} {}\n{} {}\n{}{}'.format(
        "En Yüksek Doğruluk",max(self.acc_list),
        "Ortalama Doğruluk=",statistics.mean(self.acc_list),
        "Standart Sapma=",standartSapmaBul(self.acc_list),
        "Üretilen Sonuç=",len(self.acc_list),
        "Eğitim Sayısı=",self.sayac,
        "Geçersiz Accuracy=",self.bad_acc,
        "Geçen Süre=",self.gecen,"Saniye",
        "Yapılan Tekrar=",self.tekrar,
        "Batch Size=",self.batch_size,
        "Maks Epoch",self.max_epoch,
        "Kullanılan Optimizer=",self.optimizer,
        "Sonuçlar= ",self.acc_list)
      
      
    def sonucYazdir(self):
        print(self.cikti)
    def sonucKaydet(self):
        f = open("Sonuçlar.txt", "a",encoding="utf-8")
        f.write(self.cikti+"\n\n")
    def listeDondur(self):
        return self.acc_list            
        
    def en_iyi_batch(self):
        sonuc=[]
        for i in range(1,self.batch_size+1):
            bas=time.time()               
            a=egitim(optimizer=self.optimizer,tekrar=self.tekrar,min_acc=self.min_acc,batch_size=i,max_epoch=self.max_epoch)
            a.calis()
            a=a.listeDondur()
            a=statistics.mean(a)
            son=time.time()
            gecen=round(son-bas,1)
            sonuc.append([i,a,gecen])
        for i in sonuc:      
            print("Batch Size= "+str(i[0])+
                " Accuary= "+str(i[1])+
                " Saniye= "+str(i[2]))
        
