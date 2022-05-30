# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:30:28 2020

@author: soojin
"""

import numpy as np
import scipy.stats as stats
import datetime



class KDE:
    
    def __init__(self, data): 
        
        data = data.values
        self.data = data
        
        date = data[:, 0]        
        start_time = datetime.datetime.strptime(str(data[0][0]), '%Y-%m-%d %H:%M:%S')
        
        time = list()
        time2 = list() # 5일더 붙여줄려고 한거임 통합과제에서는 안씀!!!
        
        time3 = list()

        
        for d in range(len(date)):
            p_time = datetime.datetime.strptime(str(date[d]), '%Y-%m-%d %H:%M:%S')
            time.append(int((p_time - start_time).days) + 1)
            time2.append(int((p_time - start_time).days) + 1)
            time3.append(int((p_time - start_time).days) + 1)
            
        
        for t in range(181,186):
            time2.append(t)

        for a in range(len(date)+1, len(date)+2):
            time3.append(a)
            


        self.time = time
        self.time2 = time2
        self.time3 = time3

        
        self.data_ = self.data[:, 1:]
        
        
    
    def NW(self, x, X, Y, b, kernel): # x : the number of bins / X : 실제 데이터 / b : bandwidth 
        
        if kernel == 'uniform':
            K = stats.uniform
        elif kernel == 'norm':
            K = stats.norm
            
        y_hat = list()
        
        if len(x) > 1:
            for xi in x:
                kx = list()
                for Xi in X:
                    kx.append(K.pdf(((Xi - xi)/b)))
                sum_ = np.sum(kx) / len(X)
                W = kx / (sum_ + 1e-9)          
                y_hat_ = np.matmul(np.transpose(W), Y) / len(X)
                y_hat.append(y_hat_)
        
        else:
            kx = list()
            for Xi in X:
                kx.append(K.pdf(((Xi - x)/b)))
            sum_ = np.sum(kx) / len(X)
            W = kx / (sum_ + 1e-9)       
            y_hat_ = np.matmul(np.transpose(W), Y) / len(X)
            y_hat.append(y_hat_)
            
        return y_hat
    
    

    
    
    def silverman(self):
        
        var_x = np.sqrt(np.var(self.time))
        
        def IQR(Z):
            q75, q25 = np.percentile(self.time, [75, 25])
            return q75 - q25
        
        iqr = IQR(self.time)
        m = np.min([var_x, iqr/1.349])
        
        h = (0.9 * m) / (len(self.time) ** (1 / 5))
        
        return h
    
    
    # jackknife cross-validation 방법
    def jackknife(self, bandwidth, kernel):
        
        X = self.time
        Y = self.data_[:, -1] #Y = self.data_[:, 1] 원래 이건데 데이터길이 1일때 안먹어서 바꿈
        
        pred = list()
        cross_validation = list()
        for b in bandwidth:
            pred_y = list()
            for i in range(len(X)):
                X_ = np.delete(X, i)
                Y_ = np.delete(Y, i)#np.delete(Y, i, axis = 0)
                x = [X[i]]
                pred_y.extend(self.NW(x, X_, Y_, b, kernel))
            p_y = np.array(pred_y).reshape(len(X), 1)
            Y = Y.reshape(len(Y), 1)
            pred.append(pred_y)
            
            cv_ = np.sum(np.square(Y - p_y)) / (len(X) + 1E-8)
    
            cross_validation.append([b, cv_])
            print('.', end = '')
            
        return np.array(cross_validation), pred
        
    

            
    
    
    
    # jackknife 방법으로 bandwidth 추정
    def b_jackknife(self, start, step): #start : 후보 bandwith을 설정하기 위한 첫번째 값
        
        print('find optimal jackknife value...', end = "")
        
        b_silverman = self.silverman()
        bandwidth_list = np.linspace(start, b_silverman + 1, step) 
        # silverman 방법이 보통 더 smooth하게 추정되기 때문에, silverman보다 1만큼 큰 범위 내에서 찾도록 설정함.
        cv, y_pred = self.jackknife(bandwidth_list, 'norm')
        b_jackknife = cv[np.nanargmin(cv[:, 1]), 0] # cv가 가장 작은 bandwidth 찾기
        
        print("Done.")
        
        return b_jackknife
    



    
    """ kde에서 optimal bins 결정하기 """
    def mse(self, y, pred_y):
        
        return np.sum(np.square(y - pred_y)) / len(y)
    
    
    # mse가 제일 작은 구간 찾기 (제일 작은 구간을 bins로 채택)
    def num_bins(self, b_jackknife, start, stop, step): # start : 후보 bins를 설정하기 위한 첫번째 값, stop : 마지막 값
        
        print("find optimal number of bins...", end = "")
        
        Z = self.time
        Y = self.data_
        
        xr = np.arange(start, stop, step)
        
        x_mse = list()
        for xr_ in xr:
            
            sum_ = 0.
            for i in range(Y.shape[1]):
                
                x = np.linspace(Z[0], Z[-1], xr_)
                
                y_hat_jack = self.NW(x = x, X = Z, Y = Y[:, i], b = b_jackknife, kernel = 'norm')
                y_pred_jack = self.NW(x = Z, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm')
                
                sum_ += self.mse(Y[:, i], y_pred_jack)
                
            mse_ = sum_ / len(Y)
            x_mse.append(mse_)
            print('.', end = '')
        
        print("Done.")
        
        return xr[np.nanargmin(x_mse)]
    
    # nadaraya-watson 방법으로 추정하는 함수
    #논문버젼
    def NW_predict(self, num_bins, b_jackknife): # num_bins, b_jackknife : 위에서 구한 optimal value 넣기
        
        print("make nadaraya-watson trend data...", end = "")
        
        Z = self.time
        Y = self.data_
        
        Z2 = self.time2 
        time2 = np.linspace(1, int(Z2[-1]), int(Z2[-1]))
            
        x = np.linspace(int(Z[0]), int(Z[-1]) + 1, num_bins)
        pred_y = list()
#        b_jackknife = 40
        for i in range(Y.shape[1]):
            y_hat_jack = self.NW(x = x, X = Z, Y = Y[:, i], b = b_jackknife, kernel = 'norm')
            y_pred_jack = self.NW(x = time2, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm') #논문버전
            #y_pred_jack = self.NW(x = self.time, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm') #원래버전
            pred_y.append(y_pred_jack)
            
        pred_y = np.array(pred_y)
        NW_data = np.transpose(pred_y)
        
        return NW_data
    
    #원래 버젼
    def NW_predict_original(self, num_bins, b_jackknife): # num_bins, b_jackknife : 위에서 구한 optimal value 넣기
        
        print("make nadaraya-watson trend data...", end = "")
        
        Z = self.time
        Y = self.data_
        
            
        x = np.linspace(int(Z[0]), int(Z[-1]) + 1, num_bins)
        pred_y = list()
#        b_jackknife = 40
        for i in range(Y.shape[1]):
            y_hat_jack = self.NW(x = x, X = Z, Y = Y[:, i], b = b_jackknife, kernel = 'norm')            
            y_pred_jack = self.NW(x = self.time, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm') #원래버전
            pred_y.append(y_pred_jack)
            
        pred_y = np.array(pred_y)
        NW_data_o = np.transpose(pred_y)
        
        return NW_data_o


    #논문버전_시험
    def NW_predict_test(self, num_bins, b_jackknife): # num_bins, b_jackknife : 위에서 구한 optimal value 넣기
        
        print("make nadaraya-watson trend data...", end = "")
        
        Z = self.time
        Y = self.data_
        
        Z3 = self.time3 
        time3 = np.linspace(1, int(Z3[-1]), int(Z3[-1]))
        
        
        x = np.linspace(int(Z[0]), int(Z[-1]) + 1, num_bins)
        pred_y = list()
#        b_jackknife = 40
        for i in range(Y.shape[1]):
            y_hat_jack = self.NW(x = x, X = Z, Y = Y[:, i], b = b_jackknife, kernel = 'norm')
            y_pred_jack = self.NW(x = time3, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm') #논문버전
            #y_pred_jack = self.NW(x = self.time, X = x, Y = y_hat_jack, b = b_jackknife, kernel = 'norm') #원래버전
            pred_y.append(y_pred_jack)
            
        pred_y = np.array(pred_y)
        NW_data = np.transpose(pred_y)
        
        return NW_data


    
    # raw data + nadaraya-watson 방법을 이용하여 결측값을 보간한 데이터를 만드는 함수
    def NW_interpolation(self, NW_data): # data : raw data(yc_0116) / NW_data : NW method로 추정한 데이터
        
        data = self.data
        
        nw_data = list()
        nw_intp_data = list()
        start_time = datetime.datetime.strptime(str(data[0][0]), '%Y-%m-%d %H:%M:%S')
        
        for i in range(len(NW_data)):
            nw_tmp = list()
            intp_tmp = list()
            p_time = start_time + datetime.timedelta(days = i)
            p_time = datetime.datetime.strftime(p_time, '%Y-%m-%d %H:%M:%S')
            nw_tmp.append(p_time)
            intp_tmp.append(p_time)
            
            for di in NW_data[i]:
                nw_tmp.append(di)            
            
            intp = False
            
            for d in data:
                if str(d[0]) == p_time:  #여기 d[0] -> str(d[0])로 변경
                    for di in d[1:]:
                        intp_tmp.append(di)
                    intp = True
                    
            if not intp:
                for di in NW_data[i]:
                    intp_tmp.append(di)
            
            nw_intp_data.append(intp_tmp)
            nw_data.append(nw_tmp)
                
        print("Done.")      
          
        return nw_data, nw_intp_data
    






