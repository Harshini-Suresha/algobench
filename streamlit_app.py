import numpy as np
import pandas as pd
import streamlit as st
from algobench import SUITES, UNITS, suite, exponents, theory_fit

st.set_page_config(page_title="AlgoBench", page_icon="⏱️", layout="wide")
st.title("⏱️ AlgoBench")
st.caption("Design & Analysis of Algorithms · Theory vs empirical performance")
st.markdown("[🌐 Website](https://harshini-suresha.github.io/algobench/) · [💻 GitHub](https://github.com/Harshini-Suresha/algobench) · [📓 Colab](https://colab.research.google.com/github/Harshini-Suresha/algobench/blob/main/AlgoBench.ipynb)")

tab1, tab2 = st.tabs(["Benchmark", "Complexity curves"])

with tab1:
    name = st.selectbox("Benchmark suite", SUITES)
    log = st.checkbox("Log scale (y)", value=True)
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Running..."):
            df = suite(name)
        wide = df.pivot(index="n", columns="algo", values="value")
        c1, c2 = st.columns([2, 1])
        c1.line_chart(np.log10(wide.clip(lower=1e-4)) if log else wide)
        c1.caption(("log10 of " if log else "") + UNITS[name])
        c2.subheader("Empirical exponent")
        c2.dataframe(exponents(df))
        c2.subheader("Theory fit (R²)")
        c2.dataframe(theory_fit(df))
        st.subheader("Raw data")
        st.dataframe(wide.round(4))
        st.download_button("Download CSV", df.to_csv(index=False), "algobench.csv")

with tab2:
    n = np.arange(1, 21)
    st.line_chart(pd.DataFrame({"log n": np.log2(n), "n": n, "n log n": n * np.log2(n),
                                "n^2": n**2, "2^n": 2.0**n}, index=n).clip(upper=400))
    st.markdown("Master theorem: `T(n)=aT(n/b)+f(n)`. Merge sort `a=2,b=2,f=n` → Θ(n log n).")
