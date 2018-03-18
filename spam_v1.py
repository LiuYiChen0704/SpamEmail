import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neural_network import MLPClassifier
import os
datadir = 'data/'
def get_data():
    with open(datadir+'../stopwords.txt',encoding='utf-8')as fin:
        words=fin.readline().strip()
        stopwords=words.split(',')
    emailid_tr = []
    emailtype_tr = []
    vocab_fre_total_tr = {}
    vocab_fre_list_tr = []
    with open(datadir + 'train', 'r')as fin:
        for line in fin.readlines():
            arr = line.strip().split()
            emailid_tr.append(arr[0])
            emailtype_tr.append(arr[1])
            w_f = {}
            for i in range(2, len(arr), 2):
                if arr[i].isdigit() or arr[i] in stopwords:
                    continue
                if arr[i] in vocab_fre_total_tr:
                    vocab_fre_total_tr[arr[i]] += int(arr[i + 1])
                else:
                    vocab_fre_total_tr[arr[i]] = int(arr[i + 1])
                w_f[arr[i]] = int(arr[i + 1])
            vocab_fre_list_tr.append(w_f)

    emailid_te = []
    emailtype_te = []
    vocab_fre_total_te = {}
    vocab_fre_list_te = []
    with open(datadir + 'test', 'r')as fin:
        for line in fin.readlines():
            arr = line.strip().split()
            emailid_te.append(arr[0])
            emailtype_te.append(arr[1])
            w_f = {}
            for i in range(2, len(arr), 2):
                if arr[i].isdigit() or arr[i] in stopwords:
                    continue
                if arr[i] in vocab_fre_total_te:
                    vocab_fre_total_te[arr[i]] += int(arr[i + 1])
                else:
                    vocab_fre_total_te[arr[i]] = int(arr[i + 1])
                w_f[arr[i]] = int(arr[i + 1])
            vocab_fre_list_te.append(w_f)

    word2list_tr = {}
    for word, fre in vocab_fre_total_tr.items():
        tmp = []
        for email in vocab_fre_list_tr:
            if word in email:
                tmp.append(email[word])
            else:
                tmp.append(0)
        word2list_tr[word] = tmp

    return emailid_tr,emailtype_tr,vocab_fre_total_tr,\
           vocab_fre_list_tr,emailid_te,emailtype_te,vocab_fre_total_te,vocab_fre_list_te,word2list_tr


def getEntropy(emailtype):
    p_spam=np.sum(list(map(lambda x:x=='spam',emailtype)))/len(emailtype)
    p_ham = np.sum(list(map(lambda x: x == 'ham', emailtype))) / len(emailtype)
    entropy=-p_spam*np.log(p_spam)-p_ham*np.log(p_ham)
    return entropy

def getGain(ti,word2list_tr,emailtype,entropy):
    ti_doc_list=list(map(lambda x:x>0,word2list_tr[ti]))
    ti_doc=np.sum(ti_doc_list)

    pti=ti_doc/len(word2list_tr[ti])
    spam_list=list(map(lambda x:x=='spam',emailtype))
    spam_ti_list=list(map(lambda x,y:x&y,ti_doc_list,spam_list))
    p_spam_ti=np.sum(spam_ti_list)/ti_doc

    ham_list = list(map(lambda x: x == 'ham', emailtype))
    ham_ti_list = list(map(lambda x, y: x & y, ti_doc_list, ham_list))
    p_ham_ti = np.sum(ham_ti_list) / ti_doc

    #not ti
    no_ti_doc_list = list(map(lambda x: x == 0, word2list_tr[ti]))
    no_ti_doc = np.sum(no_ti_doc_list)

    p_no_ti = no_ti_doc / len(word2list_tr[ti])
    spam_no_ti_list = list(map(lambda x, y: x & y, no_ti_doc_list, spam_list))
    p_spam_no_ti = np.sum(spam_no_ti_list) / no_ti_doc

    ham_no_ti_list = list(map(lambda x, y: x & y, no_ti_doc_list, ham_list))
    p_ham_no_ti = np.sum(ham_no_ti_list) / no_ti_doc

    if p_spam_ti!=0 and p_ham_ti!=0:
        result=pti*(-p_spam_ti*np.log(p_spam_ti)-p_ham_ti*np.log(p_ham_ti))
    elif p_spam_ti==0 and p_ham_ti!=0:
        result=pti*(-p_ham_ti*np.log(p_ham_ti))
    elif p_spam_ti!=0 and p_ham_ti==0:
        result=pti*(-p_spam_ti*np.log(p_spam_ti))
    else:
        result=0

    if p_spam_no_ti!=0 and p_ham_no_ti!=0:
        result+=p_no_ti*(-p_spam_no_ti*np.log(p_spam_no_ti)-p_ham_no_ti*np.log(p_ham_no_ti))
    elif p_spam_no_ti==0 and p_ham_no_ti!=0:
        result+=p_no_ti*(-p_ham_no_ti*np.log(p_ham_no_ti))
    elif p_spam_no_ti!=0 and p_ham_no_ti==0:
        result+=p_no_ti*(-p_spam_no_ti*np.log(p_spam_no_ti))
    else:
        result+=0

    result=entropy-result

    return result



#args:anemail:{'we':13,'told':4} features:['hello','goodby'...]
#return is a vector
def get_tf_idf(anemail,features,idf_dict):
    result=[]
    wordsum=np.sum(list(map(lambda x:x[1],anemail.items())))

    for feature in features:
        if feature in anemail:
            tf=anemail[feature]/wordsum
        else:
            tf=0
        tf_idf=tf*idf_dict[feature]
        result.append(tf_idf)
    return result

def getidf(emailid_tr,word2docs_tr,features):
    docsum = len(emailid_tr)
    idf_dict={}
    for feature in features:
        idf = docsum / word2docs_tr[feature]
        idf = np.log(idf)
        idf_dict[feature]=idf
    return idf_dict

#出现关键字的文档数
def getword2docs(word2list_tr,features):
    word2docs={}
    for feature in features:
        docs=np.sum(list(map(lambda x: x > 0, word2list_tr[feature])))
        word2docs[feature]=docs
    return word2docs

def Multi_layer_Perceptron_classifier():
    model = MLPClassifier(hidden_layer_sizes=(10, 20), activation='relu', solver='adam', alpha=0.001, batch_size=100,
                          learning_rate='adaptive', learning_rate_init=0.001, max_iter=200, shuffle=True,
                          random_state=99, tol=1e-4,
                          early_stopping=True, validation_fraction=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    return model

def logistic():
    model = LogisticRegression(n_jobs=2, random_state=98, penalty='l2')
    return model

def svm():
    model = SVC(kernel='linear')
    return model

def main():
    train_test_data='train_test.npz'
    if train_test_data not in os.listdir('.'):
        emailid_tr, emailtype_tr, vocab_fre_total_tr, \
        vocab_fre_list_tr, emailid_te, emailtype_te, vocab_fre_total_te, vocab_fre_list_te, word2list_tr=get_data()
        vocab_tr=list(map(lambda x:x[0],word2list_tr.items()))
        word_ig_tr=[]
        #train data
        for word in vocab_tr:
            gain=getGain(word,word2list_tr,emailtype_tr,getEntropy(emailtype_tr))
            word_ig_tr.append((word,gain))
        word_ig_tr_sorted=sorted(word_ig_tr,key=lambda x:x[1],reverse=True)
        features=word_ig_tr_sorted[:int(0.8*len(word_ig_tr_sorted))]
        features=[i[0] for i in features]

        X_tr=[]
        word2docs_tr=getword2docs(word2list_tr,features)
        idf_dict=getidf(emailid_tr,word2docs_tr,features)
        for anemail in vocab_fre_list_tr:
            X_tr.append(get_tf_idf(anemail,features,idf_dict))
        X_tr = np.asarray(X_tr)
        y_tr=[0 if i=='ham' else 1 for i in emailtype_tr]

        X_train, X_dev, y_train, y_dev=train_test_split(X_tr,y_tr,test_size=0.2,random_state=99)


        X_te=[]
        for anemail in vocab_fre_list_te:
            X_te.append(get_tf_idf(anemail,features,idf_dict))
        X_te = np.asarray(X_te)
        y_te=[0 if i=='ham' else 1 for i in emailtype_te]

        np.savez(train_test_data, X_train, X_dev, y_train, y_dev,X_te,y_te)
    else:
        X_train, X_dev, y_train, y_dev,X_te,y_te=np.load(train_test_data)['arr_0'],np.load(train_test_data)['arr_1'],np.load(train_test_data)['arr_2'],\
                                                 np.load(train_test_data)['arr_3'],np.load(train_test_data)['arr_4'],np.load(train_test_data)['arr_5']

    # model.fit(X_train,y_train)
    # y_pred_train = model.predict(X_train)
    # y_pred_dev=model.predict(X_dev)
    # print(classification_report(y_train, y_pred_train,target_names=['ham','spam']))
    # print(classification_report(y_dev,y_pred_dev,target_names=['ham','spam']))


    model=Multi_layer_Perceptron_classifier()
    X_train=np.concatenate((X_train,X_dev),axis=0)
    y_train=np.concatenate((y_train,y_dev),axis=0)
    #print(X_train.shape,y_train.shape)
    #print(X_te.shape,y_te.shape)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    print(classification_report(y_train, y_pred_train, target_names=['ham', 'spam']))
    y_pred_te = model.predict(X_te)
    print(classification_report(y_te, y_pred_te, target_names=['ham', 'spam']))



if __name__ == '__main__':
    main()