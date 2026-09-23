import pandas as pd
import streamlit as st
import plotly.express as px
from autopilot.config import load_models, load_routing
from autopilot.db import recent, stats
from autopilot.metrics import savings
from autopilot.settings import get_settings

st.set_page_config(page_title="LLM Cost Autopilot",layout="wide")
st.title("LLM Cost Autopilot")
st.caption("Routing, quality verification, escalation and cost analytics")
s=get_settings(); st.sidebar.write(f"Environment: `{s.app_env}`")
try: st.button("Refresh")
except Exception: pass

models=load_models(s.models_config_path)
routing=load_routing(s.routing_config_path)
stt=stats()
rows=recent(5000)

# --- Hero: the headline metric - routed cost vs "everything on the premium model" ---
baseline_name=(routing.get("metrics",{}) or {}).get("baseline_model")
baseline=models.get(baseline_name) or max(models.values(), key=lambda m: m.input_cost_per_1k+m.output_cost_per_1k)
sav=savings(rows, baseline) if rows else {"actual_cost":0.0,"baseline_cost":0.0,"savings":0.0,"savings_pct":0.0}

st.subheader("Cost savings")
h1,h2,h3,h4=st.columns(4)
h1.metric("ACTUAL COST", f"${sav['actual_cost']:.4f}")
h2.metric("BASELINE COST", f"${sav['baseline_cost']:.4f}")
h3.metric("SAVED", f"${sav['savings']:.4f}")
h4.metric("COST REDUCTION", f"{sav['savings_pct']:.2f}%")
st.caption(
    f"Baseline = every request on the highest-tier model (`{baseline.name}` / {baseline.model_id}). "
    f"Prices from config/models.yaml, pricing updated {baseline.pricing_updated_at}. "
    "Provider prices change - re-verify before publishing these numbers."
)

c1,c2,c3,c4=st.columns(4)
c1.metric("Requests",stt["requests"])
c2.metric("LLM cost",f"${stt['cost_usd']:.4f}")
c3.metric("Escalation rate",f"{stt['escalation_rate']*100:.1f}%")
c4.metric("Avg latency",f"{stt['avg_latency_ms']:.0f} ms")

threshold=float((routing.get("tiers",{}).get("tier_2",{}) or {}).get("quality_threshold",
                (routing.get("verification",{}) or {}).get("quality_threshold",4.0)))
quality_col=st.columns(3)
quality_col[0].metric("Avg quality score", f"{stt['avg_quality']:.2f} / 5.0" if stt["avg_quality"] is not None else "n/a")
if stt.get("verified_requests"):
    try:
        from autopilot.db import connect
        with connect() as _c:
            _total=_c.execute("SELECT COUNT(*) n FROM requests WHERE quality_score IS NOT NULL").fetchone()["n"]
            _ok=_c.execute("SELECT COUNT(*) n FROM requests WHERE quality_score >= ?",(threshold,)).fetchone()["n"]
        quality_col[1].metric("Quality parity", f"{_ok/_total*100:.1f}%" if _total else "n/a",
                              help=f"Verified requests at/above the quality threshold ({threshold})")
    except Exception:
        quality_col[1].metric("Quality parity", "n/a")
else:
    quality_col[1].metric("Quality parity", "n/a", help=f"Verified requests at/above the quality threshold ({threshold})")
quality_col[2].metric("Provider fallbacks", stt.get("fallbacks",0))

df=pd.DataFrame(rows)
if df.empty:
    st.info("No requests yet. Start the API and send a few requests.")
else:
    a,b=st.columns(2)
    with a:
        st.subheader("Requests by model")
        counts=df.groupby("routed_model").size().reset_index(name="count")
        st.plotly_chart(px.pie(counts,names="routed_model",values="count"),use_container_width=True)
    with b:
        st.subheader("Cost over time")
        df["time"]=pd.to_datetime(df["timestamp"],unit="s")
        daily=df.groupby(df["time"].dt.floor("h"),as_index=False)["cost_usd"].sum()
        st.plotly_chart(px.line(daily,x="time",y="cost_usd"),use_container_width=True)
    e,f=st.columns(2)
    with e:
        st.subheader("Requests by complexity tier")
        tiers=df.groupby("complexity_tier").size().reset_index(name="count")
        st.plotly_chart(px.bar(tiers,x="complexity_tier",y="count"),use_container_width=True)
    with f:
        st.subheader("Quality score distribution")
        q=df.dropna(subset=["quality_score"])
        if q.empty:
            st.caption("No verified requests yet (verification runs async or no quality-mode traffic).")
        else:
            st.plotly_chart(px.histogram(q,x="quality_score",nbins=10),use_container_width=True)
    st.subheader("Recent audit trail")
    st.dataframe(df[[c for c in ["id","time","complexity_tier","routed_model","verification_mode","routing_event","cost_usd","latency_ms","quality_score","escalated","status"] if c in df.columns]],use_container_width=True)
