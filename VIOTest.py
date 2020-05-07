import cv2
import numpy as np
from matplotlib import pyplot as plt


def run_sequence(frames,K,num_pts):
    config = np.eye(4)
    hist = [config]
    c = 0
    for l_frame,c_frame in zip(frames[:-1],frames[1:]):
        print('------------')
        print("ITER: ",str(c))
        c+=1
        

        image1 = cv2.imread(l_frame)
        image2 = cv2.imread(c_frame)
        
        orb = cv2.ORB_create()

        kpt1,des1 = orb.detectAndCompute(image1, None)
        kpt2,des2 = orb.detectAndCompute(image2, None)
                    
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1,des2)
    
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        
        pts1,pts2 = construct_pt_lists(matches,len(matches),kpt1,kpt2)
        
        
        
        E, mask = cv2.findEssentialMat(np.int32(pts1),np.int32(pts2),K,cv2.RANSAC)
        # E = K.T@F@K

        if np.linalg.det(E) < 0:
            E = -E
        
        p1,p2 = get_test_points(pts1,pts2,mask,K)
        R,t = get_rotation_translation(E,p1,p2)
        print(t)
        print(R)
        trans = get_homo_trans(R,t)
        config = config@trans
        hist.append(config)
        
    return np.array(hist)
            
def get_rotation_translation(E,pt1,pt2):
    #E = E/E[2,2]
    Eu,Es,Evh = np.linalg.svd(E)

    W = np.zeros((3,3))
    W[0,1] = -1
    W[1,0] = 1   
    W[2,2] = 1
    
    W_inv = W.T
    
    
    #need to test 
    
    R1,t1 = construct_transform(W,W,Eu,Evh)
    R2,t2 = construct_transform(W,W_inv,Eu,Evh)
    R3,t3 = construct_transform(W_inv,W,Eu,Evh)
    R4,t4 = construct_transform(W_inv,W_inv,Eu,Evh)
    
    d1_f,d1_b = get_depth(R1,t1,pt1,pt2)
    d2_f,d2_b = get_depth(R2,t2,pt1,pt2)
    d3_f,d3_b = get_depth(R3,t3,pt1,pt2)
    d4_f,d4_b = get_depth(R4,t4,pt1,pt2)

    print(d1_f,d2_f,d3_f,d4_f)
    print(d1_b,d2_b,d3_b,d4_b)
    print('sol1: ',t1,R1)
    print('sol2: ',t2,R2)
    print('sol3: ',t3,R3)
    print('sol4: ',t4,R4)
    if d1_f>0 and d1_b>0:
        print('Chose 1')
        return R1,t1
    
    if d2_f>0 and d2_b>0:
        print('Chose 2')
        return R2,t2
    
    if d3_f>0 and d3_b>0:
        print('Chose 3')
        return R3,t3
    
    if d4_f>0 and d4_b>0:
        print('Chose 4')
        return R4,t4
    
    
def get_homo_trans(R,t):
    ret = np.eye(4)
    ret[:3,:3] = R
    ret[:3,3] =t
    return ret
    
def get_depth(R,t,pt1,pt2):
    r1 = R[0,:]
    r2 = R[1,:]
    r3 = R[2,:]
    
    depth1 = np.dot(r1-pt2[0]*r3,t)/np.dot(r1 - pt2[0]*r3,pt1)
    depth2 = np.dot(r2-pt2[1]*r3,t)/np.dot(r2 - pt2[1]*r3,pt1)
    d1 = pt1*(depth1+depth2)/2
    d2 =  np.dot(R.T, d1) - np.dot(R.T, t)
    return d1[2],d2[2]
        
        
def construct_pt_lists(matches, num,kp1,kp2):
    pt1 = []
    pt2 = []
    for m in matches[:num]:
        pt1.append(kp1[m.queryIdx].pt)
        pt2.append(kp2[m.trainIdx].pt)
    return pt1,pt2

def construct_transform(rot,rot_t, U, V_t):
    R = U@rot.T@V_t
    T = U@rot_t@np.diag([1,1,0])@U.T
    t = np.array([T[2,1],-T[2,0],T[1,0]])
    return R,t

def pixel_to_cam(px,K):
    p = np.array([px[0],px[1],1])
    return np.linalg.inv(K)@p

def plot_path(hist):
    plt.plot(hist[:,0,3],hist[:,2,3])
    plt.show()
    
def get_test_points(pts1,pts2,mask,K):
    p1= np.array(pts1)[mask.flatten() ==1 ][0]
    p2 = np.array(pts2)[mask.flatten()==1][0]
    return pixel_to_cam(p1,K),pixel_to_cam(p2,K)

K = np.array([[9.842439e+02, 0.000000e+00, 6.900000e+02],[ 0.000000e+00, 9.808141e+02, 2.331966e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]])
K_inv = np.linalg.inv(K)

frame_paths = ["KITTI_data1/image_00/data/"+str(i).zfill(10)+".png" for i in np.arange(20)*3]

hist = run_sequence(frame_paths,K,100)
plot_path(hist)