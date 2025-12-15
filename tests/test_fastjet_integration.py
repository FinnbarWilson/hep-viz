import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add package root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hep_viz.data_processor import DataProcessor

try:
    import fastjet
    HAS_FASTJET = True
except ImportError:
    HAS_FASTJET = False

class TestFastJetIntegration(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataProcessor with in-memory data
        self.particles = pd.DataFrame({
            'event_id': [0, 0, 0],
            'particle_id': [1, 2, 3],
            'pdg_id': [211, 211, 22],
            'px': [10.0, 10.1, -10.0],
            'py': [0.0, 0.1, 0.0],
            'pz': [0.0, 0.0, 0.0],
            'energy': [10.0, 10.1, 10.0] # approx massless
        })
        
        self.processor = DataProcessor({
            'particles': self.particles.to_dict('records')
        })

    def test_truth_jet_clustering(self):
        if not HAS_FASTJET:
            print("Skipping FastJet test: fastjet module not found (Test passed trivially)")
            return

        result = self.processor.process_event("0")
        
        self.assertIn("jets", result)
        self.assertIn("truth", result["jets"])
        
        truth_jets = result["jets"]["truth"]
        
        # Expecting roughly 2 jets?
        # Jet 1: P1 + P2 (px~20.1, py~0.1) -> pt ~ 20.1
        # Jet 2: P3 (px~-10) -> pt ~ 10
        
        self.assertGreaterEqual(len(truth_jets), 2)
        
        pts = sorted([j['pt'] for j in truth_jets], reverse=True)
        self.assertAlmostEqual(pts[0], 20.1, delta=1.0)
        self.assertAlmostEqual(pts[1], 10.0, delta=1.0)
        
        print(f"Found {len(truth_jets)} truth jets with pTs: {pts}")

    def test_reco_jet_placeholder(self):
        # Test that reco jets logic runs without crashing even if empty
        result = self.processor.process_event("0")
        self.assertIn("reco", result["jets"])
        # Should be empty as we didn't provide tracks/calo
        self.assertEqual(len(result["jets"]["reco"]), 0)

if __name__ == '__main__':
    unittest.main()
