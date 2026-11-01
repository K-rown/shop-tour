"""Estimate each pano's true vertical (the stick leans, so panos are tilted) from vertical edges.

Vertical 3D lines are great circles through the up vector, so we vote for the up vector that the most edges agree on.
Writes frames/up.json: one [x, y, z] per point (camera frame: x right, y up, z forward).
Usage: python level_panos.py [--frames frames]
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, json
from imutil import imread, parallel_jobs

def estimate_up(path, W=1600, max_tilt=30):
    img=imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(W,W//2),interpolation=cv2.INTER_AREA).astype(np.float32)
    img=cv2.GaussianBlur(img,(0,0),1.2)
    h,w=img.shape
    gx=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
    mag=np.hypot(gx,gy)
    v,u=np.mgrid[0:h,0:w].astype(np.float32)
    th=(u+0.5)/w*2*np.pi-np.pi; ph=np.pi/2-(v+0.5)/h*np.pi
    ok=(mag>np.percentile(mag,93))&(np.abs(ph)<np.radians(62))&(v<0.78*h)
    th,ph,gx,gy,mag=th[ok],ph[ok],gx[ok],gy[ok],mag[ok]
    p=np.stack([np.cos(ph)*np.sin(th),np.sin(ph),np.cos(ph)*np.cos(th)],1)
    # unit vectors along +u (east) and up-in-image(north) at each point
    e=np.stack([np.cos(th),np.zeros_like(th),-np.sin(th)],1)
    nn=np.stack([-np.sin(ph)*np.sin(th),np.cos(ph),-np.sin(ph)*np.cos(th)],1)
    # edge tangent is perpendicular to the gradient; gradient in (east, south) pixel axes -> (east, -north)
    gE=gx; gN=-gy
    tE,tN=-gN,gE
    tau=e*tE[:,None]+nn*tN[:,None]
    tau/= (np.linalg.norm(tau,axis=1,keepdims=True)+1e-9)
    n=np.cross(p,tau); n/=(np.linalg.norm(n,axis=1,keepdims=True)+1e-9)
    wgt=np.cos(ph)*np.minimum(mag,np.percentile(mag,99))
    # candidate up vectors within cone
    best=None
    def score(U):
        d=n@U.T                      # (N,K)
        return (wgt[:,None]*np.exp(-(d/0.035)**2)).sum(0)
    cands=[]
    for tx in np.arange(-max_tilt,max_tilt+0.1,2):
        for tz in np.arange(-max_tilt,max_tilt+0.1,2):
            if np.hypot(tx,tz)>max_tilt: continue
            cands.append((tx,tz))
    def mk(tx,tz):
        a,b=np.radians(tx),np.radians(tz)
        u=np.array([np.sin(a),np.cos(a)*np.cos(b),np.sin(b)*np.cos(a)]); return u/np.linalg.norm(u)
    U=np.array([mk(*c) for c in cands]); s=score(U); k=int(np.argmax(s)); c0=cands[k]
    # refine
    cands2=[(c0[0]+dx,c0[1]+dz) for dx in np.arange(-2,2.01,0.5) for dz in np.arange(-2,2.01,0.5)]
    U2=np.array([mk(*c) for c in cands2]); s2=score(U2); k2=int(np.argmax(s2)); u=U2[k2]
    conf=float(s2[k2]/np.median(s))
    return u,conf



def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", default="frames"); a = ap.parse_args()
    pts = json.load(open(os.path.join(a.frames, "points.json")))
    with ThreadPoolExecutor(max_workers=min(parallel_jobs(1.0, 8), os.cpu_count() or 1)) as ex:     # panos are independent; numpy/cv2 release the GIL
        res = list(ex.map(lambda p: estimate_up(os.path.join(a.frames, "points", p["filename"])), pts))
    ups = []
    for p, (u, c) in zip(pts, res):
        ups.append([round(float(x), 5) for x in u])
        print(p["id"], "tilt %.1f deg" % np.degrees(np.arccos(u[1])), "conf %.1f" % c, flush=True)
    json.dump(ups, open(os.path.join(a.frames, "up.json"), "w"))
    print("wrote up.json")


if __name__ == "__main__":
    main()
