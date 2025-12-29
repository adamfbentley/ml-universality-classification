"""Quick time-dependence test"""
import numpy as np
import sys
sys.path.insert(0, '.')

from feature_extraction import FeatureExtractor
from physics_simulation import GrowthModelSimulator
from additional_surfaces import AdditionalSurfaceGenerator
from anomaly_detection import UniversalityAnomalyDetector

print('='*60, flush=True)
print('TIME-DEPENDENCE QUICK TEST', flush=True)
print('='*60, flush=True)

L, T = 64, 60
n = 5
times = [15, 30, 45, 60]

extractor = FeatureExtractor()

# Generate trajectories
print('Generating trajectories...', flush=True)
ew_trajs, kpz_trajs, qkpz_trajs = [], [], []
for i in range(n):
    sim = GrowthModelSimulator(width=L, height=T, random_state=i)
    ew_trajs.append(sim.generate_trajectory('edwards_wilkinson'))
    kpz_trajs.append(sim.generate_trajectory('kpz_equation'))
    gen = AdditionalSurfaceGenerator(width=L, height=T, random_state=i+100)
    qkpz, _ = gen.generate_quenched_kpz_surface()
    qkpz_trajs.append(qkpz)
print('Done generating.', flush=True)

# Train at T=60
print('Training detector at T=60...', flush=True)
train_X = []
for traj in ew_trajs + kpz_trajs:
    train_X.append(extractor.extract_features(traj))
train_X = np.array(train_X)

detector = UniversalityAnomalyDetector(method='isolation_forest')
detector.fit(train_X, np.array([0]*n + [1]*n))
print('Training done.', flush=True)

# Test at each time
print(flush=True)
print('Results:', flush=True)
print('Time   EW_score  KPZ_score  QKPZ_score  QKPZ_det%', flush=True)
print('-' * 55, flush=True)
for t in times:
    ew_feats = [extractor.extract_features(traj[:t]) for traj in ew_trajs]
    kpz_feats = [extractor.extract_features(traj[:t]) for traj in kpz_trajs]
    qkpz_feats = [extractor.extract_features(traj[:t]) for traj in qkpz_trajs]
    
    _, ew_scores = detector.predict(np.array(ew_feats))
    _, kpz_scores = detector.predict(np.array(kpz_feats))
    is_anom, qkpz_scores = detector.predict(np.array(qkpz_feats))
    
    print(f'{t:4d}   {np.mean(ew_scores):8.3f}  {np.mean(kpz_scores):9.3f}  {np.mean(qkpz_scores):10.3f}  {np.mean(is_anom)*100:8.0f}%', flush=True)

print(flush=True)
print('Interpretation:', flush=True)
print('- Higher score = less anomalous', flush=True)
print('- Known classes (EW, KPZ) should have increasing scores over time', flush=True)
print('- Unknown class (QKPZ) should remain flagged as anomalous', flush=True)
print('DONE', flush=True)
