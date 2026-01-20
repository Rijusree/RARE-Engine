import pandas as pd
import numpy as np

# ---------------- LOAD DATA ----------------
mpi = pd.read_csv("MPI 2025.CSV").rename(columns={'State/UT': 'state'})
pop = pd.read_csv("population growth.csv").rename(columns={'State/UT': 'state'})
upi = pd.read_csv("upi transaction.csv").rename(columns={'State/UT': 'state'})
ntl = pd.read_csv("NTL 2025.csv").rename(columns={'State/UT': 'state'})
centers = pd.read_csv("total aadhaar centers 2025.csv").rename(columns={'State/UT': 'state'})
power = pd.read_csv("power consumption 2025.csv").rename(columns={'State/UT': 'state'})

# ---------------- CLEAN STATES ----------------
def clean(df):
    df['state'] = (
        df['state']
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r'\(UT\)| UT$', '', regex=True)
    )
    return df

dfs = [mpi, pop, upi, ntl, centers, power]
master = clean(dfs[0])
for d in dfs[1:]:
    master = master.merge(clean(d), on='state', how='outer')

# ---------------- NORMALIZATION ----------------
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

# ---------------- CORE METRICS ----------------
master['upi_per_capita'] = master['Monthly_Volume_Millions'] / master['Population_2025_Millions']
master['centers_per_million'] = master['Active_Centers_Est'] / master['Population_2025_Millions']

# MBU capacity estimate (centers × avg MBU throughput)
master['mbu_capacity_estimate'] = master['Active_Centers_Est'] * 500  # 500 MBUs/center/month
master['mbu_per_capita'] = master['mbu_capacity_estimate'] / (master['Population_2025_Millions'] * 1_000_000)

# ---------------- INFRASTRUCTURE GAP INDEX (Not Ghost-Gap) ----------------
# Since we have no historical data, we measure CURRENT mismatch:
# High digital demand + Low infrastructure capacity = High risk

master['upi_norm'] = norm(master['upi_per_capita'])
master['infra_norm'] = norm(master['mbu_per_capita'])

# Infrastructure Gap Score (0-1)
# High when digital usage is high BUT infrastructure is weak
master['infrastructure_gap_score'] = (
    0.7 * master['upi_norm'] +           # Digital demand weight
    0.3 * (1 - master['infra_norm'])     # Infrastructure deficit weight
).clip(0, 1).round(4)

# Threshold-based flagging
infra_gap_threshold = master['infrastructure_gap_score'].quantile(0.70)  # Top 30%

master['infrastructure_gap_flag'] = (
    master['infrastructure_gap_score'] > infra_gap_threshold
).astype(int)

# ---------------- INTERPRETABLE ANALYSIS ----------------
def analyze_infrastructure_gap(row):
    score = row['infrastructure_gap_score']
    flag = row['infrastructure_gap_flag']
    upi = row['upi_per_capita']
    centers = row['centers_per_million']
    
    if flag == 1:
        return f"🚨 CRITICAL GAP: High digital demand ({upi:.2f} UPI/capita) with only {centers:.1f} centers/million"
    elif score >= 0.60:
        return "⚠️ WARNING: Infrastructure struggling to meet digital demand"
    elif score >= 0.40:
        return "⚡ MODERATE: Emerging capacity constraints"
    else:
        return "✅ ADEQUATE: Infrastructure matches current demand"

master['infrastructure_narrative'] = master.apply(analyze_infrastructure_gap, axis=1)

# ---------------- GHOST-GAP RISK PROXY ----------------
# Without historical data, we create a PROXY using cross-sectional signals:
# States with high UPI but low infrastructure are AT RISK of becoming Ghost-Gaps

# Find states with UPI in top 40% but infrastructure in bottom 40%
upi_high_threshold = master['upi_per_capita'].quantile(0.60)
infra_low_threshold = master['mbu_per_capita'].quantile(0.40)

master['ghost_gap_risk_proxy'] = (
    (master['upi_per_capita'] > upi_high_threshold) &
    (master['mbu_per_capita'] < infra_low_threshold)
).astype(int)

# Continuous risk score
master['ghost_gap_risk_score'] = (
    0.6 * norm(master['upi_per_capita']) +
    0.4 * (1 - norm(master['mbu_per_capita']))
).clip(0, 1).round(4)

def analyze_ghost_risk(row):
    proxy = row['ghost_gap_risk_proxy']
    score = row['ghost_gap_risk_score']
    
    if proxy == 1:
        return "🔮 HIGH GHOST-GAP RISK: Monitor for rapid UPI growth vs stagnant MBU updates"
    elif score >= 0.65:
        return "👁️ WATCH: Potential future Ghost-Gap if trends continue"
    else:
        return "📊 STABLE: No immediate Ghost-Gap risk indicators"

master['ghost_gap_risk_narrative'] = master.apply(analyze_ghost_risk, axis=1)

# ---------------- DIGITAL MATURITY SCORE ----------------
master['mpi_inverse'] = 1 - norm(master['Headcount_Ratio_2025_Percent'])

master['dms'] = (
    0.35 * norm(master['upi_per_capita']) +
    0.25 * norm(master['mbu_per_capita']) +
    0.20 * master['mpi_inverse'] +
    0.20 * norm(master['Radiance_nW_cm2_sr'])
).round(4)

# ---------------- INTERVENTION PRIORITY ----------------
# Prioritize states with infrastructure gaps + poverty
master['intervention_priority'] = (
    0.4 * master['infrastructure_gap_score'] +
    0.3 * master['ghost_gap_risk_score'] +
    0.3 * (1 - master['mpi_inverse'])  # Higher poverty = higher priority
).clip(0, 1).round(4)

# ---------------- SAVE ----------------
output_cols = [
    'state', 'Population_2025_Millions',
    'upi_per_capita', 'mbu_per_capita', 'centers_per_million',
    'infrastructure_gap_score', 'infrastructure_gap_flag', 'infrastructure_narrative',
    'ghost_gap_risk_proxy', 'ghost_gap_risk_score', 'ghost_gap_risk_narrative',
    'dms', 'intervention_priority'
]

master[output_cols].to_csv("fused_rare_engine1_2025.csv", index=False)

# ---------------- ANALYSIS OUTPUT ----------------
print("=" * 90)
print("RARE ENGINE 2025 – INFRASTRUCTURE GAP & GHOST-GAP RISK ANALYSIS")
print("(Note: Ghost-Gap detection requires historical data. Current analysis shows RISK indicators)")
print("=" * 90)

print(f"\n📊 DATASET SUMMARY:")
print(f"   Total States/UTs: {len(master)}")
print(f"   Avg UPI per capita: {master['upi_per_capita'].mean():.2f}")
print(f"   Avg Centers per million: {master['centers_per_million'].mean():.2f}")

print(f"\n🚨 INFRASTRUCTURE GAP (Current Capacity Mismatch):")
print(f"   States flagged: {master['infrastructure_gap_flag'].sum()}")
print(f"   Gap threshold: {infra_gap_threshold:.4f}")
gap_states = master[master['infrastructure_gap_flag'] == 1].sort_values('infrastructure_gap_score', ascending=False)
if len(gap_states) > 0:
    print(gap_states[['state', 'infrastructure_gap_score', 'upi_per_capita', 'centers_per_million']])
else:
    print("   No states flagged")

print(f"\n🔮 GHOST-GAP RISK PROXY (Predictive Indicators):")
print(f"   High-risk states: {master['ghost_gap_risk_proxy'].sum()}")
print(f"   UPI threshold: >{upi_high_threshold:.2f} per capita")
print(f"   Infra threshold: <{infra_low_threshold:.6f} per capita")
risk_states = master[master['ghost_gap_risk_proxy'] == 1].sort_values('ghost_gap_risk_score', ascending=False)
if len(risk_states) > 0:
    print(risk_states[['state', 'ghost_gap_risk_score', 'ghost_gap_risk_narrative']])
else:
    print("   No high-risk states detected")

print(f"\n🎯 TOP 5 INTERVENTION PRIORITIES:")
top5 = master.nlargest(5, 'intervention_priority')[['state', 'intervention_priority', 'infrastructure_narrative', 'dms']]
print(top5)

print(f"\n⚠️ IMPORTANT NOTES:")
print("   • TRUE Ghost-Gap requires UPI growth data (compare 2022→2025)")
print("   • Current 'ghost_gap_risk' shows states VULNERABLE to future gaps")
print("   • Infrastructure gap shows CURRENT demand-supply mismatch")
print("   • Recommend collecting historical UPI/MBU data for accurate Ghost-Gap detection")

print("\n✅ Output saved to: fused_rare_engine1_2025.csv")
print("=" * 90)
