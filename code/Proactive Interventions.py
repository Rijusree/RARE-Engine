import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta

print("="*80)
print("🚀 RARE ENGINE - STAGE 4: PROACTIVE INTERVENTION ENGINE")
print("   Pre-emptive MBU Guardian + Anomaly-as-Infrastructure")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📥 Loading RARE Engine data...")
df = pd.read_csv("fused_rare_engine1_2025.csv")
df = df.fillna(0)

print(f"✅ Loaded {len(df)} states/UTs\n")

# ============================================================================
# MODULE 1: PRE-EMPTIVE MBU GUARDIAN
# ============================================================================
print("🏠 MODULE 1: PRE-EMPTIVE MBU GUARDIAN")
print("   Identifying 'Digital Orphan' households for community interventions")
print("-"*80)

class PreemptiveMBUGuardian:
    """
    Identifies households at risk of becoming 'Digital Orphans' and triggers
    community-level Trust-Building Camps instead of individual alerts.
    """
    
    def __init__(self, districts_df):
        self.districts = districts_df.copy()
        
        # Simulate household-level data (in real system, this comes from Aadhaar DB)
        self._generate_household_profiles()
    
    def _generate_household_profiles(self):
        """
        Generate synthetic household profiles for demonstration.
        In production, this would query actual Aadhaar update histories.
        """
        households = []
        
        for _, district in self.districts.iterrows():
            # Simulate households per district
            n_households = int(district['Population_2025_Millions'] * 200_000)  # ~200k households per million
            
            for i in range(min(n_households, 100)):  # Sample 100 per district for demo
                # Biometric staleness (years since last update)
                staleness = np.random.choice(
                    [1, 3, 5, 8, 10, 12, 15],
                    p=[0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02]  # Most updated recently
                )
                
                # Digital engagement score (0-1)
                engagement = max(0, min(1, np.random.normal(0.6, 0.25)))
                
                # Family size
                family_size = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.15, 0.25, 0.3, 0.15, 0.1])
                
                households.append({
                    'district': district['state'],
                    'household_id': f"{district['state'][:3]}_{i:05d}",
                    'biometric_staleness_years': staleness,
                    'digital_engagement': engagement,
                    'family_size': family_size,
                    'last_update_year': 2025 - staleness
                })
        
        self.households = pd.DataFrame(households)
        print(f"   Generated {len(self.households)} household profiles")
    
    def calculate_digital_orphan_risk(self):
        """
        Calculate Digital Orphan Risk Score (0-1) based on:
        - Biometric staleness (10+ years = high risk)
        - Digital engagement (low = high risk)
        - Family vulnerability (larger families = higher impact)
        """
        df = self.households.copy()
        
        # Component 1: Staleness risk (10+ years = 1.0)
        df['staleness_risk'] = np.clip(df['biometric_staleness_years'] / 10, 0, 1)
        
        # Component 2: Low engagement risk
        df['engagement_risk'] = 1 - df['digital_engagement']
        
        # Component 3: Family impact multiplier
        df['family_multiplier'] = np.clip(df['family_size'] / 6, 0.5, 1.5)
        
        # Combined Digital Orphan Risk
        df['digital_orphan_risk'] = (
            0.5 * df['staleness_risk'] +
            0.3 * df['engagement_risk'] +
            0.2 * (df['family_multiplier'] - 0.5)  # Normalize multiplier contribution
        ).clip(0, 1)
        
        # Flag high-risk households (threshold: 0.6)
        df['is_digital_orphan'] = (df['digital_orphan_risk'] >= 0.6).astype(int)
        
        self.households = df
        return df
    
    def identify_community_clusters(self, min_cluster_size=20):
        """
        Identify districts with clusters of Digital Orphans requiring
        community-level Trust-Building Camps (not individual SMS).
        """
        orphan_counts = self.households[self.households['is_digital_orphan'] == 1].groupby('district').size()
        
        clusters = orphan_counts[orphan_counts >= min_cluster_size].sort_values(ascending=False)
        
        cluster_details = []
        for district, count in clusters.items():
            district_households = self.households[self.households['district'] == district]
            avg_risk = district_households[district_households['is_digital_orphan'] == 1]['digital_orphan_risk'].mean()
            avg_staleness = district_households[district_households['is_digital_orphan'] == 1]['biometric_staleness_years'].mean()
            
            cluster_details.append({
                'district': district,
                'orphan_households': count,
                'avg_risk_score': avg_risk,
                'avg_staleness_years': avg_staleness,
                'intervention': 'Community Trust-Building Camp'
            })
        
        return pd.DataFrame(cluster_details)

# Run Pre-emptive Guardian
guardian = PreemptiveMBUGuardian(df)
guardian.calculate_digital_orphan_risk()
orphan_clusters = guardian.identify_community_clusters(min_cluster_size=20)

print(f"\n🚨 DIGITAL ORPHAN ANALYSIS:")
print(f"   Total Households Analyzed: {len(guardian.households):,}")
print(f"   Digital Orphans Identified: {guardian.households['is_digital_orphan'].sum():,}")
print(f"   Orphan Rate: {guardian.households['is_digital_orphan'].mean():.1%}")
print(f"   Districts Requiring Community Camps: {len(orphan_clusters)}")

print(f"\n📋 TOP 10 PRIORITY DISTRICTS FOR TRUST-BUILDING CAMPS:")
print("-"*80)
print(orphan_clusters.head(10).to_string(index=False))

# ============================================================================
# MODULE 2: ANOMALY-AS-INFRASTRUCTURE
# ============================================================================
print("\n" + "="*80)
print("🚉 MODULE 2: ANOMALY-AS-INFRASTRUCTURE")
print("   Converting saturation anomalies into migrant corridor detection")
print("-"*80)

class AnomalyInfrastructureEngine:
    """
    Reinterprets 'anomalies' (>100% saturation) as infrastructure signals.
    High-saturation zones = Migrant Transit Corridors → Auto-provision kiosks.
    """
    
    def __init__(self, districts_df):
        self.districts = districts_df.copy()
        
        # Simulate saturation data
        self._generate_saturation_data()
    
    def _generate_saturation_data(self):
        """
        Generate saturation rates (some >100% indicating migrant influx).
        """
        df = self.districts.copy()
        
        # Base saturation from existing data or simulate
        if 'saturation_rate' not in df.columns:
            df['saturation_rate'] = np.random.normal(0.75, 0.25, len(df))
        
        # Identify potential migrant hubs (high UPI, high population)
        df['migration_indicator'] = (
            df.get('upi_per_capita', 0) * 0.5 +
            np.clip(df.get('Population_2025_Millions', 1) / 50, 0, 1) * 0.5
        )
        
        # Boost saturation for top migrant hubs (simulating >100%)
        top_hubs = df.nlargest(5, 'migration_indicator').index
        for idx in top_hubs:
            df.loc[idx, 'saturation_rate'] = np.random.uniform(1.05, 1.35)  # 105-135%
        
        self.districts = df
    
    def detect_migrant_corridors(self, saturation_threshold=1.0):
        """
        Identify districts with >100% saturation as Migrant Transit Corridors.
        These are NOT fraud—they're infrastructure signals.
        """
        df = self.districts.copy()
        
        # Flag corridors
        df['is_migrant_corridor'] = (df['saturation_rate'] > saturation_threshold).astype(int)
        
        # Calculate corridor intensity (how much over 100%)
        df['corridor_intensity'] = np.maximum(0, df['saturation_rate'] - 1.0)
        
        corridors = df[df['is_migrant_corridor'] == 1].copy()
        corridors = corridors.sort_values('corridor_intensity', ascending=False)
        
        return corridors
    
    def auto_provision_kiosks(self, corridors_df):
        """
        Auto-calculate kiosk requirements for each corridor.
        
        Formula: Kiosks = (Excess_Population / 5000) + Base_Kiosks
        """
        kiosk_plan = []
        
        for _, corridor in corridors_df.iterrows():
            # Calculate excess population from saturation
            base_pop = corridor['Population_2025_Millions'] * 1_000_000
            excess_pop = base_pop * corridor['corridor_intensity']
            
            # Kiosks needed (1 per 5000 excess people)
            kiosks_needed = int(np.ceil(excess_pop / 5000)) + 2  # +2 base kiosks
            
            # Estimated deployment locations
            locations = self._suggest_deployment_sites(corridor['state'], kiosks_needed)
            
            kiosk_plan.append({
                'district': corridor['state'],
                'saturation_rate': corridor['saturation_rate'],
                'corridor_intensity': corridor['corridor_intensity'],
                'excess_population_est': int(excess_pop),
                'kiosks_required': kiosks_needed,
                'deployment_sites': locations,
                'priority': 'URGENT' if corridor['corridor_intensity'] > 0.2 else 'HIGH'
            })
        
        return pd.DataFrame(kiosk_plan)
    
    def _suggest_deployment_sites(self, district, n_kiosks):
        """Suggest strategic deployment locations."""
        site_types = ['Railway Station', 'Bus Terminal', 'Labor Hub', 'Market Area', 'Industrial Zone']
        return ', '.join(np.random.choice(site_types, min(n_kiosks, 5), replace=False))

# Run Anomaly-Infrastructure Engine
anomaly_engine = AnomalyInfrastructureEngine(df)
corridors = anomaly_engine.detect_migrant_corridors(saturation_threshold=1.0)
kiosk_plan = anomaly_engine.auto_provision_kiosks(corridors)

print(f"\n🚉 MIGRANT CORRIDOR ANALYSIS:")
print(f"   Districts Analyzed: {len(df)}")
print(f"   Migrant Corridors Detected: {len(corridors)}")

if len(kiosk_plan) > 0:
    print(f"\n📋 AUTO-PROVISIONED KIOSK DEPLOYMENT PLAN:")
    print("-"*80)
    print(kiosk_plan[['district', 'saturation_rate', 'excess_population_est', 
                      'kiosks_required', 'priority']].to_string(index=False))
else:
    print("\n✅ No saturation anomalies detected (all districts <100%)")

# ============================================================================
# VISUALIZATION: STAGE 4 DASHBOARD
# ============================================================================
print("\n📊 Creating Stage 4 visualizations...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Digital Orphan Risk Distribution
ax1 = fig.add_subplot(gs[0, :2])
risk_bins = [0, 0.3, 0.6, 0.8, 1.0]
risk_labels = ['Low', 'Medium', 'High', 'Critical']
guardian.households['risk_category'] = pd.cut(guardian.households['digital_orphan_risk'], 
                                               bins=risk_bins, labels=risk_labels)
risk_counts = guardian.households['risk_category'].value_counts()

colors_risk = ['#4caf50', '#ff9800', '#ff5722', '#b71c1c']
wedges, texts, autotexts = ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
                                     colors=colors_risk, startangle=90,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('🏠 Digital Orphan Risk Distribution\n(Across All Households)', 
              fontsize=14, fontweight='bold')

# Plot 2: Orphan Clusters by District
ax2 = fig.add_subplot(gs[0, 2])
if len(orphan_clusters) > 0:
    top_clusters = orphan_clusters.head(10).sort_values('orphan_households')
    ax2.barh(range(len(top_clusters)), top_clusters['orphan_households'],
             color='#d32f2f', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_yticks(range(len(top_clusters)))
    ax2.set_yticklabels(top_clusters['district'], fontsize=9)
    ax2.set_xlabel('Orphan Households', fontsize=10, fontweight='bold')
    ax2.set_title('📊 Top 10 Districts\nRequiring Community Camps', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No significant\nclusters detected', ha='center', va='center', fontsize=12)
    ax2.set_title('Orphan Clusters', fontsize=11, fontweight='bold')

# Plot 3: Staleness vs Engagement Risk
ax3 = fig.add_subplot(gs[1, :2])
sample = guardian.households.sample(min(500, len(guardian.households)))
scatter = ax3.scatter(sample['biometric_staleness_years'], sample['digital_engagement'],
                     c=sample['digital_orphan_risk'], s=sample['family_size']*20,
                     cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.axvline(10, color='red', linestyle='--', linewidth=2, label='10-Year Staleness Threshold')
ax3.set_xlabel('Biometric Staleness (Years)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Digital Engagement Score', fontsize=11, fontweight='bold')
ax3.set_title('🔍 Digital Orphan Risk Factors\n(Bubble size = Family size)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Orphan Risk')

# Plot 4: Migrant Corridor Detection
ax4 = fig.add_subplot(gs[1, 2])
if len(corridors) > 0:
    ax4.barh(range(len(corridors)), corridors['saturation_rate'],
             color=plt.cm.Reds(corridors['corridor_intensity']),
             edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(corridors)))
    ax4.set_yticklabels(corridors['state'], fontsize=9)
    ax4.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='100% Threshold')
    ax4.set_xlabel('Saturation Rate', fontsize=10, fontweight='bold')
    ax4.set_title('🚉 Migrant Transit\nCorridors (>100%)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(axis='x', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No corridors\ndetected', ha='center', va='center', fontsize=12)
    ax4.set_title('Migrant Corridors', fontsize=11, fontweight='bold')

# Plot 5: Kiosk Deployment Plan
ax5 = fig.add_subplot(gs[2, :])
if len(kiosk_plan) > 0:
    colors_priority = ['#d32f2f' if p == 'URGENT' else '#ff9800' for p in kiosk_plan['priority']]
    bars = ax5.barh(range(len(kiosk_plan)), kiosk_plan['kiosks_required'],
                    color=colors_priority, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax5.set_yticks(range(len(kiosk_plan)))
    ax5.set_yticklabels([f"{row['district']}\n({row['saturation_rate']:.1%})" 
                         for _, row in kiosk_plan.iterrows()], fontsize=9)
    ax5.set_xlabel('Kiosks Required', fontsize=11, fontweight='bold')
    ax5.set_title('🏗️ Auto-Provisioned Temporary Kiosk Deployment Plan', fontsize=13, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(kiosk_plan['kiosks_required']):
        ax5.text(val + 0.2, i, str(val), va='center', fontsize=10, fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'No kiosk deployments required\n(All saturation rates <100%)', 
             ha='center', va='center', fontsize=14)
    ax5.set_title('Kiosk Deployment Plan', fontsize=13, fontweight='bold')

plt.suptitle('🚀 RARE Engine Stage 4: Proactive Intervention Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('stage4_proactive_interventions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("   ✅ Saved: stage4_proactive_interventions.png")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
# Save orphan clusters
orphan_clusters.to_csv('digital_orphan_clusters.csv', index=False)
print("   ✅ Saved: digital_orphan_clusters.csv")

# Save kiosk plan
if len(kiosk_plan) > 0:
    kiosk_plan.to_csv('auto_kiosk_deployment_plan.csv', index=False)
    print("   ✅ Saved: auto_kiosk_deployment_plan.csv")

# Save comprehensive report
report = {
    'module_1_digital_orphan_guardian': {
        'total_households_analyzed': len(guardian.households),
        'digital_orphans_identified': int(guardian.households['is_digital_orphan'].sum()),
        'orphan_rate': float(guardian.households['is_digital_orphan'].mean()),
        'districts_requiring_community_camps': len(orphan_clusters),
        'intervention_strategy': 'Community Trust-Building Camps (not individual SMS)'
    },
    'module_2_anomaly_infrastructure': {
        'districts_analyzed': len(df),
        'migrant_corridors_detected': len(corridors),
        'total_kiosks_provisioned': int(kiosk_plan['kiosks_required'].sum()) if len(kiosk_plan) > 0 else 0,
        'strategy': 'Convert >100% saturation anomalies into infrastructure signals'
    },
    'timestamp': datetime.now().isoformat()
}

with open('stage4_intervention_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("   ✅ Saved: stage4_intervention_report.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ STAGE 4: PROACTIVE INTERVENTIONS COMPLETE!")
print("="*80)

print(f"\n📊 MODULE 1 SUMMARY (Pre-emptive MBU Guardian):")
print(f"   • Households analyzed: {len(guardian.households):,}")
print(f"   • Digital Orphans found: {guardian.households['is_digital_orphan'].sum():,} ({guardian.households['is_digital_orphan'].mean():.1%})")
print(f"   • Community camps needed: {len(orphan_clusters)} districts")
print(f"   • Strategy: Trust-building camps for 10+ year staleness clusters")

print(f"\n📊 MODULE 2 SUMMARY (Anomaly-as-Infrastructure):")
print(f"   • Migrant corridors detected: {len(corridors)}")
if len(kiosk_plan) > 0:
    print(f"   • Temporary kiosks auto-provisioned: {kiosk_plan['kiosks_required'].sum()}")
    print(f"   • Deployment sites: Railway stations, labor hubs, bus terminals")
    print(f"   • Strategy: Preemptive kiosk placement before migrant surges")
else:
    print(f"   • No anomalies detected (healthy saturation levels)")

print(f"\n📁 Generated Files:")
print("   1. stage4_proactive_interventions.png")
print("   2. digital_orphan_clusters.csv")
if len(kiosk_plan) > 0:
    print("   3. auto_kiosk_deployment_plan.csv")
print("   4. stage4_intervention_report.json")
print("="*80)