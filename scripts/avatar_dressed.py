"""Avatar Habillé MoSL HD — Génération Vidéo 768x768 avec Tenues Vestimentaires"""
from __future__ import annotations
import argparse, json, math, os
from typing import Dict, List, Optional, Tuple
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import imageio

# ── Résolution HD ─────────────────────────────────────────────────────────────
W, H, FPS = 768, 768, 25

KP = dict(nose=0,neck=1,r_shoulder=2,r_elbow=3,r_wrist=4,
          l_shoulder=5,l_elbow=6,l_wrist=7,r_hip=8,r_knee=9,
          r_ankle=10,l_hip=11,l_knee=12,l_ankle=13,
          r_eye=14,l_eye=15,r_ear=16,l_ear=17)

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

# ── Palettes tenues (BGR) ─────────────────────────────────────────────────────
OUTFITS: Dict[str,Dict] = {
    "casual": {
        "label":"Tenue Décontractée","description":"T-shirt blanc, jean bleu, baskets blanches",
        "bg_top":(28,24,20),"bg_bottom":(12,10,8),
        "skin":(168,198,218),"skin_dark":(128,158,178),"skin_shadow":(108,138,158),
        "lips":(105,115,165),"hair":(32,26,20),"hair_hi":(55,45,35),
        "torso_main":(232,232,232),"torso_shadow":(195,195,195),"torso_hi":(248,248,248),
        "collar":(215,215,215),"sleeve":(232,232,232),"forearm":(168,198,218),
        "pants_main":(105,72,42),"pants_shadow":(78,52,28),"pants_hi":(125,90,58),
        "shoes":(235,235,235),"shoes_sole":(185,185,185),"shoes_hi":(255,255,255),
        "belt":(28,28,28),"accent":(65,125,205),"djellaba":False,
    },
    "formal": {
        "label":"Tenue Formelle","description":"Chemise bleue, pantalon noir, chaussures noires",
        "bg_top":(32,28,24),"bg_bottom":(16,13,10),
        "skin":(168,198,218),"skin_dark":(128,158,178),"skin_shadow":(108,138,158),
        "lips":(98,108,158),"hair":(26,20,14),"hair_hi":(48,38,28),
        "torso_main":(165,105,52),"torso_shadow":(132,78,32),"torso_hi":(188,128,72),
        "collar":(205,205,205),"sleeve":(165,105,52),"forearm":(168,198,218),
        "pants_main":(48,42,36),"pants_shadow":(28,24,18),"pants_hi":(65,58,50),
        "shoes":(32,26,20),"shoes_sole":(52,46,40),"shoes_hi":(55,48,42),
        "belt":(22,18,14),"accent":(205,165,82),"djellaba":False,
    },
    "traditional": {
        "label":"Tenue Traditionnelle Marocaine","description":"Djellaba blanche brodée, babouches dorées",
        "bg_top":(38,32,18),"bg_bottom":(22,16,6),
        "skin":(168,198,218),"skin_dark":(128,158,178),"skin_shadow":(108,138,158),
        "lips":(98,108,158),"hair":(26,20,14),"hair_hi":(48,38,28),
        "torso_main":(242,240,234),"torso_shadow":(212,208,198),"torso_hi":(255,254,250),
        "collar":(205,182,122),"sleeve":(242,240,234),"forearm":(242,240,234),
        "pants_main":(242,240,234),"pants_shadow":(212,208,198),"pants_hi":(255,254,250),
        "shoes":(82,162,205),"shoes_sole":(62,132,172),"shoes_hi":(105,185,225),
        "belt":(82,142,205),"accent":(62,162,225),"djellaba":True,"embroidery":(62,162,225),
    },
}

# ── Utilitaires ───────────────────────────────────────────────────────────────
def pt(kp_arr, idx, thr=0.05):
    if kp_arr is None or idx>=len(kp_arr): return None
    c = kp_arr[idx,2] if kp_arr.shape[1]>2 else 1.0
    return (int(kp_arr[idx,0]),int(kp_arr[idx,1])) if c>=thr else None

def dist(a,b):
    return 0.0 if (a is None or b is None) else math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def midpoint(a,b):
    return None if (a is None or b is None) else ((a[0]+b[0])//2,(a[1]+b[1])//2)

def lerp(a,b,t):
    return (int(a[0]*(1-t)+b[0]*t), int(a[1]*(1-t)+b[1]*t))

def draw_limb(canvas, a, b, color, width, shadow=None, highlight=None):
    """Membre comme polygone rempli avec ombre et reflet pour effet 3D."""
    if a is None or b is None: return
    dx,dy = b[0]-a[0],b[1]-a[1]
    ln = math.sqrt(dx*dx+dy*dy) or 1
    nx,ny = -dy/ln, dx/ln
    w2 = width/2
    pts = np.array([
        [a[0]+nx*w2*1.0, a[1]+ny*w2*1.0],
        [b[0]+nx*w2*0.8, b[1]+ny*w2*0.8],
        [b[0]-nx*w2*0.8, b[1]-ny*w2*0.8],
        [a[0]-nx*w2*1.0, a[1]-ny*w2*1.0],
    ], dtype=np.int32)
    if shadow:
        sp = (pts + np.array([3,4])).astype(np.int32)
        ov = canvas.copy(); cv2.fillPoly(ov,[sp],shadow)
        cv2.addWeighted(ov,0.45,canvas,0.55,0,canvas)
    cv2.fillPoly(canvas,[pts],color)
    if highlight:
        hi_pts = np.array([
            [a[0]+nx*w2*0.6, a[1]+ny*w2*0.6],
            [b[0]+nx*w2*0.5, b[1]+ny*w2*0.5],
            [b[0]+nx*w2*0.1, b[1]+ny*w2*0.1],
            [a[0]+nx*w2*0.1, a[1]+ny*w2*0.1],
        ], dtype=np.int32)
        ov2 = canvas.copy(); cv2.fillPoly(ov2,[hi_pts],highlight)
        cv2.addWeighted(ov2,0.25,canvas,0.75,0,canvas)
    cv2.polylines(canvas,[pts],True,tuple(max(0,c-25) for c in color),1,cv2.LINE_AA)


def draw_head(canvas, body, outfit, scale=1.5):
    nose=pt(body,KP["nose"]); neck=pt(body,KP["neck"])
    if nose is None and neck is None: return
    if nose and neck: hs=int(dist(nose,neck)*1.15); cx=(nose[0]+neck[0])//2; cy=nose[1]-hs//3
    elif nose: hs=int(55*scale); cx,cy=nose[0],nose[1]-int(28*scale)
    else: hs=int(55*scale); cx,cy=neck[0],neck[1]-int(62*scale)
    hs=max(int(32*scale),min(hs,int(95*scale)))
    rx=int(hs*.74); ry=int(hs*.92)
    sk=outfit["skin"]; sd=outfit["skin_dark"]; ss=outfit["skin_shadow"]
    hr=outfit["hair"]; hh=outfit["hair_hi"]

    # Ombre portée tête
    ov=canvas.copy()
    cv2.ellipse(ov,(cx+5,cy+6),(rx+2,ry+2),0,0,360,(8,6,4),-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.4,canvas,0.6,0,canvas)

    # Cheveux arrière (volume)
    cv2.ellipse(canvas,(cx,cy-int(ry*.18)),(rx+7,ry+9),0,0,360,hr,-1,cv2.LINE_AA)
    # Reflet cheveux
    cv2.ellipse(canvas,(cx-int(rx*.2),cy-int(ry*.6)),(int(rx*.4),int(ry*.2)),0,0,360,hh,-1,cv2.LINE_AA)

    # Visage base
    cv2.ellipse(canvas,(cx,cy),(rx,ry),0,0,360,sk,-1,cv2.LINE_AA)

    # Ombre latérale droite (volume 3D)
    ov2=canvas.copy()
    shadow_pts=np.array([[cx+int(rx*.5),cy-ry],[cx+rx+3,cy-int(ry*.3)],
                          [cx+rx+3,cy+int(ry*.5)],[cx+int(rx*.4),cy+ry]],dtype=np.int32)
    cv2.fillPoly(ov2,[shadow_pts],sd)
    cv2.addWeighted(ov2,0.38,canvas,0.62,0,canvas)

    # Ombre sous menton
    ov3=canvas.copy()
    cv2.ellipse(ov3,(cx,cy+int(ry*.75)),(int(rx*.6),int(ry*.18)),0,0,180,ss,-1,cv2.LINE_AA)
    cv2.addWeighted(ov3,0.35,canvas,0.65,0,canvas)

    # Cheveux dessus (forme naturelle)
    hp=np.array([[cx-rx,cy-int(ry*.35)],[cx-int(rx*.7),cy-ry-10],[cx-int(rx*.2),cy-ry-14],
                 [cx,cy-ry-16],[cx+int(rx*.2),cy-ry-14],[cx+int(rx*.7),cy-ry-10],
                 [cx+rx,cy-int(ry*.35)],[cx+int(rx*.6),cy-int(ry*.55)],
                 [cx,cy-int(ry*.58)],[cx-int(rx*.6),cy-int(ry*.55)]],dtype=np.int32)
    cv2.fillPoly(canvas,[hp],hr)
    # Reflet cheveux dessus
    cv2.ellipse(canvas,(cx-int(rx*.15),cy-int(ry*.72)),(int(rx*.28),int(ry*.12)),0,0,360,hh,-1,cv2.LINE_AA)

    # Yeux
    ey=cy-int(ry*.14); eo=int(rx*.40); erx=int(rx*.26); ery=int(ry*.14)
    for ex in [cx-eo,cx+eo]:
        # Paupière supérieure (ombre)
        cv2.ellipse(canvas,(ex,ey-1),(erx+2,ery+2),0,0,360,(20,15,10),-1,cv2.LINE_AA)
        # Blanc de l'œil
        cv2.ellipse(canvas,(ex,ey),(erx,ery),0,0,360,(245,243,240),-1,cv2.LINE_AA)
        # Iris (dégradé simulé)
        cv2.circle(canvas,(ex,ey),int(erx*.68),(70,48,28),-1,cv2.LINE_AA)
        cv2.circle(canvas,(ex,ey),int(erx*.55),(55,36,18),-1,cv2.LINE_AA)
        # Pupille
        cv2.circle(canvas,(ex,ey),int(erx*.32),(8,6,4),-1,cv2.LINE_AA)
        # Reflets
        cv2.circle(canvas,(ex-int(erx*.25),ey-int(ery*.3)),max(2,int(erx*.18)),(255,255,255),-1,cv2.LINE_AA)
        cv2.circle(canvas,(ex+int(erx*.15),ey+int(ery*.1)),max(1,int(erx*.08)),(220,220,220),-1,cv2.LINE_AA)
        # Cils supérieurs
        cv2.ellipse(canvas,(ex,ey),(erx+2,ery+2),0,195,345,(12,8,5),2,cv2.LINE_AA)
        # Cils inférieurs
        cv2.ellipse(canvas,(ex,ey),(erx+1,ery+1),0,15,165,(12,8,5),1,cv2.LINE_AA)

    # Sourcils (forme arquée naturelle)
    bry=ey-int(ry*.20); bw=int(rx*.34)
    for bx in [cx-eo,cx+eo]:
        side = -1 if bx<cx else 1
        bpts=np.array([[bx-bw,bry+3],[bx-int(bw*.3),bry-4],[bx+int(bw*.3),bry-5],
                        [bx+bw,bry+1],[bx+int(bw*.5),bry+4],[bx-int(bw*.5),bry+3]],dtype=np.int32)
        cv2.fillPoly(canvas,[bpts],hr)

    # Nez (plus détaillé)
    nby=cy+int(ry*.10); nw=int(rx*.20); ntip=cy+int(ry*.22)
    # Arête du nez
    cv2.line(canvas,(cx,cy-int(ry*.05)),(cx,ntip),sd,2,cv2.LINE_AA)
    # Bout du nez
    cv2.ellipse(canvas,(cx,ntip),(nw,int(ry*.10)),0,0,360,sk,-1,cv2.LINE_AA)
    cv2.ellipse(canvas,(cx,ntip),(nw,int(ry*.10)),0,0,360,sd,1,cv2.LINE_AA)
    # Narines
    for nxo,nyo in [(-int(nw*.62),int(ry*.04)),(int(nw*.62),int(ry*.04))]:
        cv2.ellipse(canvas,(cx+nxo,ntip+nyo),(int(nw*.32),int(ry*.07)),0,0,360,sd,-1,cv2.LINE_AA)

    # Bouche
    my=cy+int(ry*.36); mw=int(rx*.38); lp=outfit.get("lips",(105,115,165))
    lp_dark=tuple(max(0,c-30) for c in lp); lp_hi=tuple(min(255,c+25) for c in lp)
    # Lèvre supérieure (arc de cupidon)
    ul_pts=np.array([[cx-mw,my],[cx-int(mw*.4),my-int(ry*.06)],[cx,my-int(ry*.04)],
                      [cx+int(mw*.4),my-int(ry*.06)],[cx+mw,my],[cx,my+2]],dtype=np.int32)
    cv2.fillPoly(canvas,[ul_pts],lp)
    # Lèvre inférieure
    ll_pts=np.array([[cx-mw+3,my],[cx,my+int(ry*.10)],[cx+mw-3,my],[cx,my+3]],dtype=np.int32)
    cv2.fillPoly(canvas,[ll_pts],lp)
    # Reflet lèvre inférieure
    cv2.ellipse(canvas,(cx,my+int(ry*.06)),(int(mw*.35),int(ry*.04)),0,0,180,lp_hi,-1,cv2.LINE_AA)
    # Ligne de bouche
    cv2.line(canvas,(cx-mw,my),(cx+mw,my),lp_dark,1,cv2.LINE_AA)
    # Commissures
    for mx2 in [cx-mw,cx+mw]:
        cv2.circle(canvas,(mx2,my),2,lp_dark,-1,cv2.LINE_AA)

    # Oreilles
    eary=cy+int(ry*.06); earrx=int(rx*.16); earry=int(ry*.26)
    for earxp,side in [(cx-rx,-1),(cx+rx,1)]:
        cv2.ellipse(canvas,(earxp,eary),(earrx,earry),0,0,360,sk,-1,cv2.LINE_AA)
        cv2.ellipse(canvas,(earxp,eary),(int(earrx*.65),int(earry*.7)),0,0,360,sd,1,cv2.LINE_AA)
        cv2.ellipse(canvas,(earxp+side*2,eary),(int(earrx*.35),int(earry*.45)),0,0,360,ss,1,cv2.LINE_AA)

    # Cou
    if neck:
        nck_w=int(rx*.36)
        draw_limb(canvas,(cx,cy+ry-6),neck,sk,nck_w,ss)
        # Ombre cou
        ov4=canvas.copy()
        cv2.ellipse(ov4,(cx,cy+ry-2),(int(nck_w*.5),int(ry*.08)),0,0,180,ss,-1,cv2.LINE_AA)
        cv2.addWeighted(ov4,0.3,canvas,0.7,0,canvas)


def draw_hand(canvas, hand_kp, skin, skin_dark, skin_shadow):
    if hand_kp is None or len(hand_kp)<21: return
    # Paume
    palm_idx=[0,1,5,9,13,17]; palm_pts=[]
    for i in palm_idx:
        p=pt(hand_kp,i,thr=0.01)
        if p: palm_pts.append(p)
    if len(palm_pts)>=3:
        arr=np.array(palm_pts,dtype=np.int32)
        hull=cv2.convexHull(arr)
        # Ombre paume
        ov=canvas.copy(); cv2.fillPoly(ov,[(hull+np.array([2,3])).astype(np.int32)],skin_shadow)
        cv2.addWeighted(ov,0.35,canvas,0.65,0,canvas)
        cv2.fillPoly(canvas,[hull],skin)
        cv2.polylines(canvas,[hull],True,skin_dark,1,cv2.LINE_AA)
    # Doigts
    fingers=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    widths=[8,7,7,6,5]
    for finger,w in zip(fingers,widths):
        for i in range(len(finger)-1):
            a=pt(hand_kp,finger[i],thr=0.01); b=pt(hand_kp,finger[i+1],thr=0.01)
            if a and b:
                draw_limb(canvas,a,b,skin,w,skin_shadow)
                # Articulation
                cv2.circle(canvas,b,max(2,w//3-1),skin_dark,-1,cv2.LINE_AA)
    # Ongles
    for tip in [4,8,12,16,20]:
        t=pt(hand_kp,tip,thr=0.01); base=pt(hand_kp,tip-1,thr=0.01)
        if t and base:
            nail=(210,205,200)
            cv2.ellipse(canvas,t,(5,3),0,0,360,nail,-1,cv2.LINE_AA)
            cv2.ellipse(canvas,t,(5,3),0,0,360,skin_dark,1,cv2.LINE_AA)

def draw_body(canvas, body, outfit):
    djellaba=outfit.get("djellaba",False)
    neck=pt(body,KP["neck"]); rsh=pt(body,KP["r_shoulder"]); lsh=pt(body,KP["l_shoulder"])
    rel=pt(body,KP["r_elbow"]); lel=pt(body,KP["l_elbow"])
    rwr=pt(body,KP["r_wrist"]); lwr=pt(body,KP["l_wrist"])
    rhi=pt(body,KP["r_hip"]); lhi=pt(body,KP["l_hip"])
    rkn=pt(body,KP["r_knee"]); lkn=pt(body,KP["l_knee"])
    ran=pt(body,KP["r_ankle"]); lan=pt(body,KP["l_ankle"])
    tc=outfit["torso_main"]; ts=outfit["torso_shadow"]; th=outfit["torso_hi"]
    sl=outfit["sleeve"]; fa=outfit["forearm"]
    pc=outfit["pants_main"]; ps=outfit["pants_shadow"]; ph=outfit["pants_hi"]
    sc=outfit["shoes"]; ss2=outfit["shoes_sole"]; sh2=outfit["shoes_hi"]
    bc=outfit["belt"]; ac=outfit["accent"]
    sk=outfit["skin"]; sd=outfit["skin_dark"]; sss=outfit["skin_shadow"]

    if djellaba:
        if neck and rhi and lhi:
            hm=midpoint(rhi,lhi); sm=midpoint(rsh,lsh) if rsh and lsh else neck
            by=H-40
            if ran and lan: by=max(ran[1],lan[1])+20
            # Corps djellaba (trapèze évasé)
            dpts=np.array([[sm[0]-55,sm[1]],[sm[0]+55,sm[1]],
                           [hm[0]+80,by],[hm[0]-80,by]],dtype=np.int32)
            # Ombre portée
            ov=canvas.copy(); cv2.fillPoly(ov,[(dpts+np.array([6,5])).astype(np.int32)],ts)
            cv2.addWeighted(ov,0.4,canvas,0.6,0,canvas)
            cv2.fillPoly(canvas,[dpts],tc)
            # Reflet gauche
            ov2=canvas.copy()
            hi_pts=np.array([[sm[0]-55,sm[1]],[sm[0]-20,sm[1]],
                              [hm[0]-20,by],[hm[0]-80,by]],dtype=np.int32)
            cv2.fillPoly(ov2,[hi_pts],th); cv2.addWeighted(ov2,0.2,canvas,0.8,0,canvas)
            # Ombre droite
            ov3=canvas.copy()
            sh_pts=np.array([[sm[0]+20,sm[1]],[sm[0]+55,sm[1]],
                              [hm[0]+80,by],[hm[0]+20,by]],dtype=np.int32)
            cv2.fillPoly(ov3,[sh_pts],ts); cv2.addWeighted(ov3,0.3,canvas,0.7,0,canvas)
            # Broderies
            emb=outfit.get("embroidery",ac)
            # Ligne centrale
            cv2.line(canvas,(sm[0],sm[1]+25),(hm[0],by-15),emb,3,cv2.LINE_AA)
            # Motifs géométriques marocains
            for yo in range(sm[1]+50,by-25,55):
                # Losange
                pts_d=np.array([[hm[0],yo-12],[hm[0]+10,yo],[hm[0],yo+12],[hm[0]-10,yo]],dtype=np.int32)
                cv2.polylines(canvas,[pts_d],True,emb,2,cv2.LINE_AA)
                cv2.circle(canvas,(hm[0],yo),3,emb,-1,cv2.LINE_AA)
                # Lignes horizontales décoratives
                cv2.line(canvas,(hm[0]-22,yo),(hm[0]-12,yo),emb,1,cv2.LINE_AA)
                cv2.line(canvas,(hm[0]+12,yo),(hm[0]+22,yo),emb,1,cv2.LINE_AA)
        # Babouches
        for an,side in [(ran,1),(lan,-1)]:
            if an:
                sw=16
                spts=np.array([[an[0]-sw,an[1]],[an[0]+sw+side*12,an[1]],
                               [an[0]+sw+side*18,an[1]+18],[an[0]-sw+side*3,an[1]+18]],dtype=np.int32)
                ov=canvas.copy(); cv2.fillPoly(ov,[(spts+np.array([3,3])).astype(np.int32)],ss2)
                cv2.addWeighted(ov,0.4,canvas,0.6,0,canvas)
                cv2.fillPoly(canvas,[spts],sc)
                cv2.polylines(canvas,[spts],True,ss2,1,cv2.LINE_AA)
                # Broderie babouche
                cv2.ellipse(canvas,(an[0]+side*3,an[1]+9),(sw-3,5),0,0,180,ac,2,cv2.LINE_AA)
                cv2.circle(canvas,(an[0]+side*3,an[1]+4),3,ac,-1,cv2.LINE_AA)
    else:
        # Jambes
        for hi,kn,an,side in [(rhi,rkn,ran,1),(lhi,lkn,lan,-1)]:
            if hi and kn: draw_limb(canvas,hi,kn,pc,28,ps,ph)
            if kn and an: draw_limb(canvas,kn,an,pc,22,ps,ph)
            # Chaussures
            if an:
                sw=16
                spts=np.array([[an[0]-sw,an[1]],[an[0]+sw+side*9,an[1]],
                               [an[0]+sw+side*15,an[1]+16],[an[0]-sw,an[1]+16]],dtype=np.int32)
                ov=canvas.copy(); cv2.fillPoly(ov,[(spts+np.array([3,3])).astype(np.int32)],ss2)
                cv2.addWeighted(ov,0.4,canvas,0.6,0,canvas)
                cv2.fillPoly(canvas,[spts],sc)
                cv2.polylines(canvas,[spts],True,ss2,1,cv2.LINE_AA)
                cv2.line(canvas,(an[0]-sw,an[1]+16),(an[0]+sw+side*15,an[1]+16),ss2,2,cv2.LINE_AA)
                # Reflet chaussure
                ov2=canvas.copy()
                cv2.ellipse(ov2,(an[0]+side*2,an[1]+5),(sw-4,4),0,0,180,sh2,-1,cv2.LINE_AA)
                cv2.addWeighted(ov2,0.3,canvas,0.7,0,canvas)

    # Torse
    if rsh and lsh and rhi and lhi:
        tpts=np.array([[rsh[0],rsh[1]],[lsh[0],lsh[1]],[lhi[0]+6,lhi[1]],[rhi[0]-6,rhi[1]]],dtype=np.int32)
        ov=canvas.copy(); cv2.fillPoly(ov,[(tpts+np.array([4,5])).astype(np.int32)],ts)
        cv2.addWeighted(ov,0.4,canvas,0.6,0,canvas)
        cv2.fillPoly(canvas,[tpts],tc)
        # Reflet gauche
        ov2=canvas.copy()
        hi_t=np.array([[lsh[0],lsh[1]],[lsh[0]+int((rsh[0]-lsh[0])*.35),lsh[1]],
                        [lhi[0]+int((rhi[0]-lhi[0])*.35),lhi[1]],[lhi[0]+6,lhi[1]]],dtype=np.int32)
        cv2.fillPoly(ov2,[hi_t],th); cv2.addWeighted(ov2,0.22,canvas,0.78,0,canvas)
        # Ombre droite
        ov3=canvas.copy()
        sh_t=np.array([[rsh[0]-int((rsh[0]-lsh[0])*.3),rsh[1]],[rsh[0],rsh[1]],
                        [rhi[0]-6,rhi[1]],[rhi[0]-int((rhi[0]-lhi[0])*.3),rhi[1]]],dtype=np.int32)
        cv2.fillPoly(ov3,[sh_t],ts); cv2.addWeighted(ov3,0.3,canvas,0.7,0,canvas)

    # Ceinture
    if rhi and lhi and not djellaba:
        by2=(rhi[1]+lhi[1])//2; mx=(rhi[0]+lhi[0])//2
        cv2.line(canvas,(rhi[0]-6,by2),(lhi[0]+6,by2),bc,7,cv2.LINE_AA)
        cv2.rectangle(canvas,(mx-7,by2-4),(mx+7,by2+4),ac,-1)
        cv2.rectangle(canvas,(mx-7,by2-4),(mx+7,by2+4),tuple(max(0,c-30) for c in ac),1)

    # Bras
    for sh,el,wr in [(rsh,rel,rwr),(lsh,lel,lwr)]:
        if sh and el: draw_limb(canvas,sh,el,sl,22,ts,th)
        if el and wr:
            fc=fa if not djellaba else sl
            draw_limb(canvas,el,wr,fc,18,sss if not djellaba else ts)

    # Col / encolure
    col=outfit.get("collar",tc)
    if neck and rsh and lsh:
        cpts=np.array([[neck[0]-12,neck[1]+8],[neck[0]+12,neck[1]+8],
                       [lsh[0]+14,lsh[1]-6],[rsh[0]-14,rsh[1]-6]],dtype=np.int32)
        cv2.fillPoly(canvas,[cpts],col)
        if djellaba:
            emb=outfit.get("embroidery",ac)
            cv2.polylines(canvas,[cpts],True,emb,2,cv2.LINE_AA)
            cv2.circle(canvas,neck,7,emb,-1,cv2.LINE_AA)
            cv2.circle(canvas,neck,11,emb,2,cv2.LINE_AA)
            cv2.circle(canvas,neck,15,emb,1,cv2.LINE_AA)
        else:
            # Boutons chemise
            if rhi:
                for by3 in range(neck[1]+18,rhi[1] if rhi else neck[1]+100,22):
                    cv2.circle(canvas,(neck[0],by3),3,ac,-1,cv2.LINE_AA)
                    cv2.circle(canvas,(neck[0],by3),3,tuple(max(0,c-40) for c in ac),1,cv2.LINE_AA)


def draw_background(canvas, outfit, w=W, h=H):
    bt=outfit["bg_top"]; bb=outfit["bg_bottom"]
    for y in range(h):
        t=y/h; color=tuple(int(bt[c]*(1-t)+bb[c]*t) for c in range(3))
        canvas[y,:]=color
    # Halo central (lumière de fond)
    cx,cy=w//2,int(h*.38)
    Yi,Xi=np.ogrid[:h,:w]
    d=np.sqrt((Xi-cx)**2+(Yi-cy)**2).astype(np.float32)
    halo=np.clip(1.0-d/(w*.7),0,1)*.22
    for c in range(3):
        canvas[:,:,c]=np.clip(canvas[:,:,c].astype(np.float32)+halo*65,0,255).astype(np.uint8)
    # Sol
    fy=int(h*.90)
    cv2.line(canvas,(0,fy),(w,fy),tuple(min(255,c+18) for c in bb),1,cv2.LINE_AA)
    # Ombre au sol (ellipse sous les pieds)
    ov=canvas.copy()
    cv2.ellipse(ov,(w//2,fy+4),(int(w*.18),10),0,0,360,(4,3,2),-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.5,canvas,0.5,0,canvas)

def render_frame(kp_data, outfit, w=W, h=H):
    canvas=np.zeros((h,w,3),dtype=np.uint8)
    draw_background(canvas,outfit,w,h)
    body=kp_data.get("body"); hl=kp_data.get("hand_left"); hr=kp_data.get("hand_right")
    if body is None: return canvas
    draw_body(canvas,body,outfit)
    draw_head(canvas,body,outfit)
    sk=outfit["skin"]; sd=outfit["skin_dark"]; ss=outfit["skin_shadow"]
    draw_hand(canvas,hr,sk,sd,ss)
    draw_hand(canvas,hl,sk,sd,ss)
    # Vignette douce
    Yi,Xi=np.ogrid[:h,:w]
    d=np.sqrt((Xi-w//2)**2+(Yi-h//2)**2).astype(np.float32)
    vig=np.clip(1.0-d/(w*.78),0,1)[:,:,np.newaxis]
    return (canvas.astype(np.float32)*vig).astype(np.uint8)

def load_json_kp(path):
    with open(path,encoding="utf-8") as f: data=json.load(f)
    res={"body":None,"hand_left":None,"hand_right":None}
    if "people" not in data or not data["people"]: return res
    p=data["people"][0]
    def parse(flat,n):
        if not flat: return None
        return np.array(flat,dtype=np.float32).reshape(-1,3)[:n]
    res["body"]=parse(p.get("pose_keypoints_2d",[]),18)
    res["hand_left"]=parse(p.get("hand_left_keypoints_2d",[]),21)
    res["hand_right"]=parse(p.get("hand_right_keypoints_2d",[]),21)
    return res

def load_frames(json_dir):
    files=sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    if not files: raise FileNotFoundError(f"Aucun JSON dans {json_dir}")
    return [load_json_kp(os.path.join(json_dir,f)) for f in files]

def scale_keypoints(frames, src_w=512, src_h=512, dst_w=W, dst_h=H):
    """Redimensionne les coordonnées keypoints de src vers dst."""
    sx=dst_w/src_w; sy=dst_h/src_h
    for f in frames:
        for key in ["body","hand_left","hand_right"]:
            kp=f.get(key)
            if kp is not None:
                kp[:,0]*=sx; kp[:,1]*=sy
    return frames

def smooth_kp(frames, sigma=1.2):
    if len(frames)<3 or sigma<=0: return frames
    def _s(key,n):
        seq=[]; valid=[]
        for f in frames:
            kp=f.get(key)
            if kp is not None and len(kp)>=n: seq.append(kp[:n].copy()); valid.append(True)
            else: seq.append(np.zeros((n,3),dtype=np.float32)); valid.append(False)
        arr=np.stack(seq,axis=0)
        arr[:,:,0]=gaussian_filter1d(arr[:,:,0],sigma=sigma,axis=0)
        arr[:,:,1]=gaussian_filter1d(arr[:,:,1],sigma=sigma,axis=0)
        for i,(f,v) in enumerate(zip(frames,valid)):
            if v: f[key]=arr[i]
    _s("body",18); _s("hand_left",21); _s("hand_right",21)
    return frames

def smooth_px(frames, sigma=0.5):
    if len(frames)<3 or sigma<=0: return frames
    arr=np.stack(frames,axis=0).astype(np.float32)
    sm=gaussian_filter1d(arr,sigma=sigma,axis=0)
    return [np.clip(sm[i],0,255).astype(np.uint8) for i in range(len(frames))]

def save_mp4(frames, path, fps=FPS):
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    rgb=[cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames]
    wr=imageio.get_writer(path,fps=fps,codec="libx264",quality=9,pixelformat="yuv420p",macro_block_size=None)
    for f in rgb: wr.append_data(f)
    wr.close()
    print(f"  ✓ {os.path.basename(path)}  ({len(frames)} frames @ {fps:.0f}fps, {os.path.getsize(path)/1e6:.1f} MB)")

def generate(json_dir, output, outfit_name="casual", sigma_kp=1.2, sigma_px=0.5,
             fps=FPS, w=W, h=H, src_w=512, src_h=512):
    if outfit_name not in OUTFITS: raise ValueError(f"Tenue inconnue: {outfit_name}")
    outfit=OUTFITS[outfit_name]
    print(f"  Tenue   : {outfit['label']}")
    print(f"  Détails : {outfit['description']}")
    frames=load_frames(json_dir)
    print(f"  {len(frames)} frames chargées")
    # Redimensionner les coordonnées pour la résolution cible
    if w!=src_w or h!=src_h:
        frames=scale_keypoints(frames,src_w,src_h,w,h)
    if sigma_kp>0: frames=smooth_kp(frames,sigma_kp)
    rendered=[]
    for kp in tqdm(frames,desc=f"  Rendu [{outfit_name}]",unit="fr",ncols=72):
        rendered.append(render_frame(kp,outfit,w,h))
    if sigma_px>0: rendered=smooth_px(rendered,sigma_px)
    save_mp4(rendered,output,fps)

def main():
    ap=argparse.ArgumentParser(description="Avatar habillé MoSL HD",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--json-dir",default="data/processed/keypoints_2d/sample")
    ap.add_argument("--output",default=None)
    ap.add_argument("--output-dir",default="outputs/avatar_dressed")
    ap.add_argument("--outfit",choices=["casual","formal","traditional"],default="casual")
    ap.add_argument("--all-outfits",action="store_true")
    ap.add_argument("--list-outfits",action="store_true")
    ap.add_argument("--fps",type=float,default=FPS)
    ap.add_argument("--width",type=int,default=W)
    ap.add_argument("--height",type=int,default=H)
    ap.add_argument("--sigma-kp",type=float,default=1.2)
    ap.add_argument("--sigma-px",type=float,default=0.5)
    args=ap.parse_args()

    if args.list_outfits:
        print("\nTenues disponibles :")
        for k,v in OUTFITS.items(): print(f"  {k:12s} — {v['label']} : {v['description']}")
        return 0

    outfits=list(OUTFITS.keys()) if args.all_outfits else [args.outfit]
    os.makedirs(args.output_dir,exist_ok=True)
    sign=os.path.basename(os.path.normpath(args.json_dir))

    for name in outfits:
        out=args.output if (args.output and len(outfits)==1) else os.path.join(args.output_dir,f"{sign}_{name}_hd.mp4")
        print(f"\n{'='*58}\n  Génération : {name.upper()}  ({args.width}×{args.height})\n  Sortie     : {out}\n{'='*58}")
        generate(args.json_dir,out,name,args.sigma_kp,args.sigma_px,args.fps,args.width,args.height)

    print(f"\nTerminé. Vidéos dans : {args.output_dir}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
