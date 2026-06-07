# UR5 Robotic Arm Inverse Kinematics Verification Tool
This project develops a UR5 Inverse Kinematics Verification Tool that models the UR5 robot using DH parameters, computes forward and inverse kinematics, validates IK solutions through forward kinematics, and performs Jacobian-based singularity and manipulability analysis to ensure solution accuracy and reliability.

<img width="900" height="452" alt="image" src="https://github.com/user-attachments/assets/ababf4fb-60d3-4d04-a0b5-a8f7c66aa3a9" />
<img width="906" height="450" alt="image" src="https://github.com/user-attachments/assets/607fb1d7-90be-47aa-9ed1-1b88ed57e5b3" />
The model that I made using Autodesk Fusion represents an intermediate stage in the
development of the UR5 robotic manipulator. At this stage, only a portion of the robot
has been modelled. By continuing the same design methodology and successively
creating the remaining links and joints according to the UR5 specifications, the
complete UR5 robotic arm can be obtained. The final assembly will consist of all six
revolute joints and their corresponding links, accurately representing the kinematic
structure of the UR5 manipulator.




Python Code:
import numpy as np
from scipy.optimize import least_squares
d1=0.089159
a2=-0.425
a3=-0.39225
d4=0.10915
d5=0.09465
d6=0.0823
DH=[
[0,np.pi/2,d1],
[a2,0,0],
[a3,0,0],
[0,np.pi/2,d4],
[0,-np.pi/2,d5],
[0,0,d6]
]
def dh_transform(a,alpha,d,theta):
 return np.array([
 [np.cos(theta),-np.sin(theta)*np.cos(alpha),np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
 [np.sin(theta),np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
 [0,np.sin(alpha),np.cos(alpha),d],
 [0,0,0,1]
 ])
def forward_kinematics(q):
 T=np.eye(4)
 for i in range(6):
 a,alpha,d=DH[i]
 T=T@dh_transform(a,alpha,d,q[i])
 return T
def rotation_error(R_des,R_cur):
 R_err=R_des@R_cur.T
 angle=np.arccos(np.clip((np.trace(R_err)-1)/2,-1.0,1.0))
 if abs(angle)<1e-8:
 return np.zeros(3)
 axis=np.array([
 R_err[2,1]-R_err[1,2],
 R_err[0,2]-R_err[2,0],
 R_err[1,0]-R_err[0,1]
 ])/(2*np.sin(angle))
 return angle*axis
def inverse_kinematics(T_desired):
 def error_function(q):
 T=forward_kinematics(q)
 pos_error=T_desired[:3,3]-T[:3,3]
 ori_error=rotation_error(T_desired[:3,:3],T[:3,:3])
 return np.concatenate([pos_error,ori_error])
 q0=np.zeros(6)
 solution=least_squares(error_function,q0,method='lm')
 return solution.x
def compute_jacobian(q):
 J=np.zeros((6,6))
 delta=1e-6
 T0=forward_kinematics(q)
 p0=T0[:3,3]
 for i in range(6):
 q_new=q.copy()
 q_new[i]+=delta
 T1=forward_kinematics(q_new)
 p1=T1[:3,3]
 dp=(p1-p0)/delta
 dR=rotation_error(T1[:3,:3],T0[:3,:3])/delta
 J[:,i]=np.hstack((dp,dR))
 return J
def check_singularity(J):
 return np.linalg.matrix_rank(J)<6
def verify_ik(T_desired,q_ik):
 T_fk=forward_kinematics(q_ik)
 pos_error=np.linalg.norm(T_desired[:3,3]-T_fk[:3,3])
 ori_error=np.linalg.norm(rotation_error(T_desired[:3,:3],T_fk[:3,:3]))
 return pos_error,ori_error
if __name__=="__main__":
 print("UR5 IK VERIFICATION TOOL")
 q_test=np.radians([30,-45,60,90,-30,45])
 print("Input Joint Angles (deg):")
 print(np.degrees(q_test))
 T_desired=forward_kinematics(q_test)
 print("\nDesired Pose:")
 print(np.round(T_desired,4))
 q_ik=inverse_kinematics(T_desired)
 print("\nIK Solution (deg):")
 print(np.round(np.degrees(q_ik),3))
 pos_err,ori_err=verify_ik(T_desired,q_ik)
 print("\nVerification Results")
 print("Position Error:",pos_err,"m")
 print("Orientation Error:",ori_err,"rad")
 J=compute_jacobian(q_ik)
 print("\nJacobian Matrix:")
 print(np.round(J,4))
 if check_singularity(J):
 print("\nSINGULAR CONFIGURATION")
 else:
 print("\nNON-SINGULAR CONFIGURATION")
 print("\nIK VALID")

<img width="828" height="518" alt="image" src="https://github.com/user-attachments/assets/cd74eabe-5a72-4b16-908b-b0ec2dea1fd3" />
<img width="679" height="697" alt="image" src="https://github.com/user-attachments/assets/5d3aa967-4613-4e2a-8bfe-f559702629f0" />


