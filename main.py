import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# defining variables and parameters
best_threshold,EER = 0,0
threshold = [(i+5)*20 for i in range(30)] # cheking for different values of threshold
tp,fp,tn,fn = 0,0,0,0 # true positive , false positive , true negative , false negative
epochs = 150
count = 0

# defing axes for graphs
x_points = np.array(threshold)
y_frr = np.array([])
y_far = np.array([])

# getting images and file names
files = os.listdir("dl-classProject-data-15-sub/")
images = [cv2.imread(file) for file in glob.glob("dl-classProject-data-15-sub/*")]

# creating orb feature extractor
orb = cv2.ORB_create(nfeatures = 5000)

# creating brute force matcher
bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)

# looping through images
for i in range(0,len(threshold)):
    for j in range(0,epochs):
        for k in range(j,epochs):
            count+=1
            img1 = images[200+j]
            img2 = images[200+k]
            
            # resizing images
            width = int(( img1.shape[1] + img2.shape[1] )/ 2)
            height = int(( img1.shape[0] + img2.shape[0] )/ 2)
            dimensions = (width,height)
            img1 = cv2.resize(img1,dimensions,interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2,dimensions,interpolation=cv2.INTER_AREA)

            # computing keypoints and descriptors
            kp1 , desc1 = orb.detectAndCompute(img1,None)
            kp2 , desc2 = orb.detectAndCompute(img2,None)

            # brute force matching
            matches = bf.match(desc1,desc2)
            matches = sorted(matches,key = lambda x:x.distance)

            # calculating similarity
            similar_regions = [m for m in matches if m.distance<int(threshold[i])]
            similarity = len(similar_regions)/len(matches)

            # optional code to print and view matching in images
            if count%1001==0:
                print("similarity =",similarity)
            #     img1 = cv2.drawKeypoints(img1,kp1,None)
            #     img2 = cv2.drawKeypoints(img2,kp2,None)
            #     img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,img2,flags=2)
            #     cv2.imshow('img3',img3)
            #     cv2.waitKey(0) 
            #     cv2.destroyAllWindows() 

            # calculating values in confusion matrix
            pred = 1 if similarity >= 0.5 else 0
            out = 1 if files[0][2] == files[j][2] else 0

            if pred==1 and out==1:
                tp+=1
            elif pred==1 and out==0:
                fp+=1
            elif pred==0 and out==1:
                fn+=1
            else : 
                tn+=1
        
    # performance parameters
    cf_matrix = ([tn,fp],[fn,tp]) 
    print("confusion matrix ,",cf_matrix)

    far = fp/(fp+tn)
    frr = fn/(fn+tp)
    crr = (tp+tn)/(tp+tn+fp+fn)
    accuracy = 1 - ((frr+far)/2)

    y_far = np.append(y_far,far)
    y_frr = np.append(y_frr,frr)

    if(abs(far-frr)<0.07):
        best_threshold = threshold[i]
        EER = far

    print("accuracy = ",accuracy ," crr = ",crr ," with threshold = ",threshold[i])
  
print("best threshold = ",best_threshold," EER = ",EER)

# plot of EER
plt.figure(0)
plt.plot(x_points,y_far, color='b',label='FAR')
plt.plot(x_points,y_frr, color='g',label='FRR')
plt.text(best_threshold,EER,f'({best_threshold},{EER})')

plt.ylabel("Rate")
plt.xlabel("Threshold")
plt.title("EER")
plt.legend()

# Plot of ROC
plt.figure(1)
plt.plot(y_far,y_frr, color='y')
  
plt.ylabel("FRR")
plt.xlabel("FAR")
plt.title("ROC")
plt.legend()

plt.show()