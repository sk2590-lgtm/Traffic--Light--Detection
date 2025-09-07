import cv2,os,time,csv,pygame,platform,subprocess,math
import numpy as np
from datetime import datetime
from collections import deque

pygame.mixer.init()
base=os.path.join(os.getcwd(),"sounds")
def beep(f):
    p=os.path.join(base,f)
    if os.path.exists(p):
        pygame.mixer.music.load(p)
        pygame.mixer.music.play()

cap=cv2.VideoCapture(0)
if not cap.isOpened(): quit()
fourcc=cv2.VideoWriter_fourcc(*"XVID")
out=cv2.VideoWriter("output.avi",fourcc,20.0,(720,480))

rng={
 "RED":[(np.array([0,120,70]),np.array([10,255,255])),
        (np.array([170,120,70]),np.array([180,255,255]))],
 "YELLOW":[(np.array([15,120,120]),np.array([35,255,255]))],
 "GREEN":[(np.array([40,70,70]),np.array([90,255,255]))]
}

cnts={"RED":0,"YELLOW":0,"GREEN":0}
last={"RED":0,"YELLOW":0,"GREEN":0}
frames=0;hit=0
hist={"RED":deque(maxlen=6),"YELLOW":deque(maxlen=6),"GREEN":deque(maxlen=6)}
hold={"RED":0,"YELLOW":0,"GREEN":0}

f=open("detections.csv","w",newline='')
w=csv.writer(f);w.writerow(["time","state"])

pulse=0
fps=0;last_fps=time.time()

while True:
    ok,fr=cap.read()
    if not ok: break
    frames+=1
    fr=cv2.resize(fr,(720,480))
    hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV)
    cur=[]
    for c,rr in rng.items():
        msk=None
        for lo,hi in rr:
            m=cv2.inRange(hsv,lo,hi)
            msk=m if msk is None else cv2.bitwise_or(msk,m)
        # morphological cleanup for faster stable detection
        kernel = np.ones((5,5),np.uint8)
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
        msk = cv2.morphologyEx(msk, cv2.MORPH_DILATE, kernel)

        cs,_=cv2.findContours(msk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for x in cs:
            a=cv2.contourArea(x)
            if a<600: continue
            x_,y,w_,h=cv2.boundingRect(x)
            asp=w_/float(h)
            p=cv2.arcLength(x,True)
            if p==0: continue
            circ=(4*np.pi*a)/(p**2)
            if 0.65<=circ<=1.2 and 0.7<=asp<=1.3:
                cur.append((c,(x_,y,w_,h)))
    seen=set([c for c,_ in cur])
    for c in ["RED","YELLOW","GREEN"]:
        hist[c].append(1 if c in seen else 0)

    stable=[]
    now=time.time()
    for c,h in hist.items():
        if sum(h)>=2:   # faster stabilization (was 4)
            stable.append(c)
            hold[c]=now+0.2   # shorter hold (was 0.5)
        elif hold[c]>now:
            stable.append(c)

    if stable:
        hit+=1
        for s in stable:
            t=time.time()
            if t-last[s]>1:   # quicker re-detection (was 2)
                if s=="RED": beep("red_alert.wav")
                elif s=="YELLOW": beep("yellow_alert.wav")
                elif s=="GREEN": beep("green_alert.wav")
                cnts[s]+=1
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),s])
                last[s]=t

    pulse+=1
    glow=int((math.sin(pulse/8.0)+1)*100+50)

    for c,(x,y,w_,h) in cur:
        if c in stable:
            col=(0,255,0) if c=="GREEN" else ((0,255,255) if c=="YELLOW" else (0,0,255))
            cv2.rectangle(fr,(x,y),(x+w_,y+h),col,2+int(abs(math.sin(pulse/5))*3))
            cv2.putText(fr,c,(x,y-8),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)

    cv2.rectangle(fr,(0,0),(720,60),(25,25,25),-1)

    pos={"RED":(40,30),"YELLOW":(120,30),"GREEN":(200,30)}
    col={"RED":(0,0,255),"YELLOW":(0,255,255),"GREEN":(0,255,0)}
    for c,(xx,yy) in pos.items():
        if c in stable:
            cv2.circle(fr,(xx,yy),22,col[c],-1)
            cv2.circle(fr,(xx,yy),22,(255,255,255),2)
        else:
            cv2.circle(fr,(xx,yy),20,(80,80,80),-1)
            cv2.circle(fr,(xx,yy),20,col[c],2)
        cv2.putText(fr,c[0],(xx-10,yy+8),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    offx=500
    for c in ["RED","YELLOW","GREEN"]:
        cv2.circle(fr,(offx,30),12,col[c],-1)
        cv2.putText(fr,str(cnts[c]),(offx+20,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        offx+=90

    now=time.time()
    if now-last_fps>=1:
        fps=int(frames/(now-last_fps))
        last_fps=now;frames=0
    cv2.putText(fr,f"FPS: {fps}",(600,55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,255,200),2)

    cv2.imshow("Traffic Light Monitor",fr)
    out.write(fr)
    if cv2.waitKey(1)&0xFF==ord("q"): break

cap.release();out.release();f.close();cv2.destroyAllWindows()
acc=(hit/frames*100) if frames>0 else 0
rp="report.txt"
dt=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(rp,"w",encoding="utf-8") as r:
    r.write("╔══════════════════════════════════════╗\n")
    r.write("║        TRAFFIC LIGHT PROJECT         ║\n")
    r.write("╚══════════════════════════════════════╝\n\n")
    r.write(f"Total Frames Processed : {frames}\n")
    r.write(f"Frames With Detection  : {hit}\n")
    r.write(f"Overall Accuracy       : {acc:.2f} %\n\n")
    r.write("Detection Counts:\n")
    r.write(f"  🔴 Red     : {cnts['RED']}\n")
    r.write(f"  🟡 Yellow  : {cnts['YELLOW']}\n")
    r.write(f"  🟢 Green   : {cnts['GREEN']}\n\n")
    r.write(f"Report Generated On: {dt}\n")
    r.write("\n----------------------------------------\n")
    r.write("Special Thanks 🙏 to GitHub for giving us\n")
    r.write("the chance to showcase and present this project.\n")
    r.write("----------------------------------------\n")

if platform.system()=="Windows":
    os.startfile(rp)
elif platform.system()=="Darwin":
    subprocess.call(["open",rp])
else:
    subprocess.call(["xdg-open",rp])
