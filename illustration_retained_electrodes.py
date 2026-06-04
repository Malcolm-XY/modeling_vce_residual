# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:52:47 2026

@author: 18307
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

raw_pcc_03="""
NRR=0.3 (number of channels: 18)
PCC
Original	Exp.	RQ	GG	Sig.
Fp2			Fp2	
F7	F7	F7	F7	F7
F5	F5	F5	F5	F5
			F3	
F6	F6	F6	F6	F6
F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7
FC5	FC5	FC5	FC5	FC5
		FC3		
FC6	FC6	FC6	FC6	FC6
FT8	FT8	FT8	FT8	FT8
T7	T7	T7	T7	T7
C5	C5	C5	C5	C5
C3	C3	C3		C3
C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz
C2	C2	C2	C2	C2
C4	C4	C4	C4	C4
	C6		C6	C6
T8	T8	T8	T8	T8
TP7	TP7			
				
				
				CP2
"""

raw_plv_03="""
NRR=0.3 (number of channels: 18)
PLV
Original	Exp.	RQ	GG	Sig.
Fp1				
Fpz				
Fp2			Fp2	
F7	F7	F7	F7	F7
F5	F5	F5	F5	F5
			F3	
F6	F6	F6	F6	F6
F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7
FC5		FC5	FC5	FC5
				
FC6		FC6	FC6	FC6
FT8	FT8	FT8	FT8	FT8
T7	T7	T7	T7	T7
	C5	C5	C5	C5
				
C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz
C2	C2	C2	C2	C2
				
				
T8	T8	T8	T8	T8
TP7	TP7	TP7	TP7	TP7
		CP1		CP1
CPz	CPz	CPz	CPz	CPz
	CP2			
	P1	P1		P1
	Pz			

"""

raw_pcc_02="""
NRR=0.2 (number of channels: 12)
PCC
Original	Exp.	RQ	GG	Sig.
Fp2				
F7	F7	F7	F7	F7
F5	F5	F5	F5	F5
				
		F6	F6	F6
F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7
				
				
FC6	FC6	FC6	FC6	FC6
FT8	FT8		FT8	
T7	T7	T7	T7	T7
				
				
C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz
C2	C2	C2	C2	C2
	C4	C4		C4
				
T8	T8	T8	T8	T8
"""

raw_plv_02="""
NRR=0.2 (number of channels: 12)
PLV
Original	Exp.	RQ	GG	Sig.
Fp2				
F7	F7	F7	F7	F7
F5		F5	F5	F5
				
			F6	
F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7
				
				
			FC6	
FT8	FT8	FT8	FT8	FT8
T7	T7	T7	T7	T7
				
				
C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz
	C2			
				
				
T8	T8	T8	T8	T8
TP7	TP7	TP7		TP7
				
CPz	CPz	CPz	CPz	CPz
				
	P1	P1		P1
"""

# 
raw = """
NRR=0.3 (number of channels: 18)
PCC	PLV
Original	Exp.	RQ	GG	Sig.	Original	Exp.	RQ	GG	Sig.
					Fp1				
					Fpz				
Fp2			Fp2		Fp2			Fp2	
F7	F7	F7	F7	F7	F7	F7	F7	F7	F7
F5	F5	F5	F5	F5	F5	F5	F5	F5	F5
			F3					F3	
F6	F6	F6	F6	F6	F6	F6	F6	F6	F6
F8	F8	F8	F8	F8	F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7
FC5	FC5	FC5	FC5	FC5	FC5		FC5	FC5	FC5
		FC3							
FC6	FC6	FC6	FC6	FC6	FC6		FC6	FC6	FC6
FT8	FT8	FT8	FT8	FT8	FT8	FT8	FT8	FT8	FT8
T7	T7	T7	T7	T7	T7	T7	T7	T7	T7
C5	C5	C5	C5	C5		C5	C5	C5	C5
C3	C3	C3		C3					
C1	C1	C1	C1	C1	C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz
C2	C2	C2	C2	C2	C2	C2	C2	C2	C2
C4	C4	C4	C4	C4					
	C6		C6	C6					
T8	T8	T8	T8	T8	T8	T8	T8	T8	T8
TP7	TP7				TP7	TP7	TP7	TP7	TP7
							CP1		CP1
					CPz	CPz	CPz	CPz	CPz
				CP2		CP2			
						P1	P1		P1
						Pz			
"""

raw_ = """
NRR=0.3 (number of channels: 18)
PCC	PLV
Original	Exp.	RQ	GG	Sig.	Original	Exp.	RQ	GG	Sig.						
									
Fp2					Fp2				
F7	F7	F7	F7	F7	F7	F7	F7	F7	F7
F5	F5	F5	F5	F5	F5		F5	F5	F5
									
		F6	F6	F6				F6	
F8	F8	F8	F8	F8	F8	F8	F8	F8	F8
FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7	FT7
									
									
FC6	FC6	FC6	FC6	FC6				FC6	
FT8	FT8		FT8		FT8	FT8	FT8	FT8	FT8
T7	T7	T7	T7	T7	T7	T7	T7	T7	T7
									
									
C1	C1	C1	C1	C1	C1	C1	C1	C1	C1
Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz	Cz
C2	C2	C2	C2	C2		C2			
	C4	C4		C4					
									
T8	T8	T8	T8	T8	T8	T8	T8	T8	T8
					TP7	TP7	TP7		TP7
									
					CPz	CPz	CPz	CPz	CPz
									
						P1	P1		P1			
"""

# Put longer prefixes before shorter ones to avoid parsing "Fpz" as "F" + "pz".
prefix_order = ["Fp", "AF", "FT", "FC", "TP", "CP", "PO", "F", "T", "C", "P", "O"]
tokens = re.findall(r'\b(?:Fp|AF|FT|FC|TP|CP|PO|F|T|C|P|O)(?:z|\d{1,2})\b', raw_pcc_02)

def split_eeg_label(label):
    for pref in prefix_order:
        if label.startswith(pref):
            suffix = label[len(pref):]
            return pref, suffix
    return label, ""

anterior_order = ["Fp", "AF", "F", "FT", "FC", "T", "C", "TP", "CP", "P", "PO", "O"]

def lateral_key(suffix):
    if suffix == "z":
        return 0
    n = int(suffix)
    # left odd channels first, midline, then right even channels within each row
    # use signed position: odd -> negative, even -> positive
    signed = -n if n % 2 == 1 else n
    return signed

unique_chs = sorted(
    set(tokens),
    key=lambda ch: (
        anterior_order.index(split_eeg_label(ch)[0]),
        lateral_key(split_eeg_label(ch)[1]),
        ch
    )
)

# Load standard montage positions from MNE.
try:
    import mne
    montage = mne.channels.make_standard_montage("standard_1005")
    pos3d = montage.get_positions()["ch_pos"]
    xy = {ch: np.array([pos3d[ch][0], pos3d[ch][1]]) for ch in unique_chs if ch in pos3d}
    missing = [ch for ch in unique_chs if ch not in pos3d]
except Exception as e:
    xy = {}
    missing = unique_chs[:]
    print("MNE montage could not be loaded:", e)

if missing:
    print("Missing channels in montage:", missing)

fig, ax = plt.subplots(figsize=(8, 9), dpi=220)

# Head outline
theta = np.linspace(0, 2*np.pi, 600)
radius = 0.105
ax.plot(radius*np.cos(theta), radius*np.sin(theta), linewidth=1.6)

# Nose and ears
ax.plot([-0.018, 0, 0.018], [radius*0.98, radius*1.14, radius*0.98], linewidth=1.6)
ear_h = 0.046
ear_w = 0.018
t = np.linspace(-np.pi/2, np.pi/2, 160)
for s in [-1, 1]:
    ax.plot(s*(radius + ear_w*np.cos(t)), ear_h*np.sin(t), linewidth=1.3)

# Scatter selected electrodes
counts = {ch: tokens.count(ch) for ch in unique_chs}
plot_chs = [ch for ch in unique_chs if ch in xy]
xs = [xy[ch][0] for ch in plot_chs]
ys = [xy[ch][1] for ch in plot_chs]
sizes = [20 + 15 * counts[ch] for ch in plot_chs]

sc = ax.scatter(xs, ys, s=sizes, edgecolors="black", linewidths=0.8, alpha=0.9, zorder=3)

# Add labels
from adjustText import adjust_text
texts = []
for ch in plot_chs:
    x, y = xy[ch]
    texts.append(
        ax.text(
            x, y + 0.0048, ch,
            ha="center", va="bottom",
            fontsize=16,
            zorder=4
        )
    )

adjust_text(
    texts,
    ax=ax,
    expand_text=(1.2, 1.3),
    expand_points=(1.3, 1.4),
    force_text=(0.4, 0.6),
    force_points=(0.3, 0.5),
    arrowprops=dict(arrowstyle="-", lw=0.5)
)

ax.set_title("Electrodes Involved in NRR=0.2, PCC", fontsize=20, pad=18)
ax.text(
    0, -0.120,
    f"Unique electrodes: {len(unique_chs)}", # | Marker size roughly reflects occurrence frequency",
    ha="center", va="center", fontsize=20
)

ax.set_aspect("equal")
ax.set_xlim(-0.135, 0.135)
ax.set_ylim(-0.145, 0.135)
ax.axis("off")
