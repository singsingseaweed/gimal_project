import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

st.set_page_config(
    page_title="제어공학기말",
)

st.header("제어공학기말 202021025 임수창")
st.subheader("2번문제 streamlit 으로 공유")


num = [100]
den = [1, 5, 106]

ltf = signal.TransferFunction(num, den)

zeros, poles, _ = signal.tf2zpk(num, den)

t = np.linspace(0, 5, 1000)
u = np.ones_like(t)

t, y, _ = signal.lsim(ltf, u, t)

fig1, ax1 = plt.subplots()
ax1.plot(t, y)
ax1.set(xlabel='Time', ylabel='Output', title='Step Response')
ax1.grid(True)
st.pyplot(fig1)

w, mag, phase = signal.bode(ltf)

fig2, (ax2, ax3) = plt.subplots(2, 1)
ax2.semilogx(w, mag)
ax2.set(xlabel='Frequency [rad/s]', ylabel='Magnitude [dB]', title='Bode Plot - Magnitude')
ax2.grid(True)

ax3.semilogx(w, phase)
ax3.set(xlabel='Frequency [rad/s]', ylabel='Phase [degrees]', title='Bode Plot - Phase')
ax3.grid(True)

fig2.tight_layout()
st.pyplot(fig2)

fig3, ax4 = plt.subplots()
ax4.plot(np.real(zeros), np.imag(zeros), 'o', label='Zeros')
ax4.plot(np.real(poles), np.imag(poles), 'x', label='Poles')
ax4.set(xlabel='Real', ylabel='Imaginary', title='Pole-Zero Plot')
ax4.grid(True)
ax4.legend()

st.pyplot(fig3)