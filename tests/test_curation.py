"""
Unit tests for peptide curation, physicochemical feature extraction, and splitting logic.
Uses Python's standard library unittest for complete portability.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.curation import PeptideCurator
from src.features.physicochemical import PhysicochemicalExtractor
from src.splitting.homology_split import SequenceClusterer


class TestCurationAndFeatures(unittest.TestCase):

    def test_peptide_curator_valid_sequence(self):
        curator = PeptideCurator(min_length=10, max_length=50)
        valid_seq = "GLFDIVKKVVGALGSL"
        is_valid, reason = curator.is_valid_sequence(valid_seq)
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_peptide_curator_invalid_length(self):
        curator = PeptideCurator(min_length=10, max_length=50)
        short_seq = "ACDEF"
        is_valid, reason = curator.is_valid_sequence(short_seq)
        self.assertFalse(is_valid)
        self.assertIn("min_length", reason)

    def test_peptide_curator_non_canonical_amino_acid(self):
        curator = PeptideCurator(min_length=10, max_length=50)
        invalid_seq = "GLFDIVKKXBZALGSL"  # contains X, B, Z
        is_valid, reason = curator.is_valid_sequence(invalid_seq)
        self.assertFalse(is_valid)
        self.assertIn("Non-canonical", reason)

    def test_physicochemical_extractor_values(self):
        seq = "KKKKKKKKKK"  # Pure poly-lysine: strong positive charge at pH 7
        charge = PhysicochemicalExtractor.calculate_net_charge(seq, ph=7.0)
        self.assertGreater(charge, 8.0)

        seq_neutral = "AAAAAAAAAA"  # Poly-alanine: non-polar, neutral
        features = PhysicochemicalExtractor.extract_sequence_features(seq_neutral)
        self.assertIn("length", features)
        self.assertEqual(features["length"], 10.0)
        self.assertEqual(features["aac_A"], 1.0)
        self.assertEqual(features["aac_K"], 0.0)

    def test_sequence_clusterer_identical_sequences(self):
        seqs = ["GLFDIVKKVVGALGSL", "GLFDIVKKVVGALGSL", "AAAAAAAAAA"]
        clusters = SequenceClusterer.cluster_by_identity(seqs, identity_threshold=0.5)
        # The two identical sequences should belong to the exact same cluster
        self.assertEqual(clusters[0], clusters[1])
        # And different from the poly-alanine
        self.assertNotEqual(clusters[0], clusters[2])


if __name__ == "__main__":
    unittest.main()

