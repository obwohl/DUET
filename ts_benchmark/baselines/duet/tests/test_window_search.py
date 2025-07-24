import unittest
import numpy as np
import pandas as pd
import sys
import os

# Fügt das Projekt-Stammverzeichnis zum Python-Pfad hinzu, um den Import zu ermöglichen
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

from ts_benchmark.baselines.duet.utils.window_search import find_interesting_windows

class TestWindowSearch(unittest.TestCase):

    def setUp(self):
        """Definiert gemeinsame Parameter für die Tests."""
        self.seq_len = 96
        self.horizon = 24
        # Der Offset, ab dem die Suche beginnen darf: seq_len - horizon
        self.search_offset = 72
        self.data_len = 300
        np.random.seed(42)

    def test_output_structure_and_types(self):
        """Testet, ob die Ausgabe die korrekte Struktur und die richtigen Datentypen hat."""
        data = pd.DataFrame(np.random.rand(self.data_len, 2), columns=['A', 'B'])
        results = find_interesting_windows(data, self.horizon, self.seq_len)

        self.assertIsInstance(results, dict, "Die Ausgabe sollte ein Dictionary sein.")
        self.assertIn('A', results)
        self.assertIn('B', results)

        channel_a_results = results['A']
        self.assertIsInstance(channel_a_results, dict)
        self.assertIn('max_mean_diff_idx', channel_a_results)
        self.assertIn('max_var_diff_idx', channel_a_results)
        self.assertIn('max_trend_rev_idx', channel_a_results)
        self.assertIn('max_ks_dist_idx', channel_a_results)
        
        # Überprüft, ob die Indizes als Integer zurückgegeben werden
        self.assertIsInstance(channel_a_results['max_mean_diff_idx'], int)
        self.assertIsInstance(channel_a_results['max_ks_dist_idx'], int)

    def test_handles_input_too_short_for_search(self):
        """
        Testet, ob die Funktion ein leeres Dict zurückgibt, wenn die Daten zu kurz sind,
        um auch nur ein durchsuchbares Fensterpaar zu bilden.
        Die minimale Länge ist `seq_len + horizon`.
        """
        # Die Daten sind genau einen Zeitschritt zu kurz
        short_data = pd.DataFrame(np.random.rand(self.seq_len + self.horizon - 1, 1), columns=['A'])
        results = find_interesting_windows(short_data, self.horizon, self.seq_len)
        self.assertEqual(results, {}, "Sollte ein leeres Dict zurückgeben, wenn die Daten kürzer als seq_len + horizon sind.")

    def test_handles_flat_series_correctly(self):
        """
        Dies ist der Kerntest für den Bugfix.
        Bei einer flachen Zeitreihe sind alle Metriken 0. `np.argmax` für ein Array aus Nullen gibt den Index 0 zurück.
        Die Funktion muss `0 + search_offset` zurückgeben, nicht nur 0.
        """
        flat_data = pd.DataFrame(np.ones((self.data_len, 1)), columns=['flat_channel'])
        results = find_interesting_windows(flat_data, self.horizon, self.seq_len)

        # Das "interessanteste" Fenster in den durchsuchten Daten (die alle flach sind) ist bei Index 0.
        # Die Funktion muss den Offset zu diesem Index addieren.
        expected_index = self.search_offset
        
        self.assertIn('flat_channel', results)
        channel_results = results['flat_channel']
        
        self.assertEqual(channel_results['max_mean_diff_idx'], expected_index, "Index für flache Serie (Mittelwert) ist falsch.")
        self.assertEqual(channel_results['max_var_diff_idx'], expected_index, "Index für flache Serie (Varianz) ist falsch.")
        self.assertEqual(channel_results['max_trend_rev_idx'], expected_index, "Index für flache Serie (Trend) ist falsch.")
        self.assertEqual(channel_results['max_ks_dist_idx'], expected_index, "Index für flache Serie (KS-Distanz) ist falsch.")

    def test_ignores_windows_before_search_offset(self):
        """
        Testet, dass ein sehr offensichtliches Ereignis vor dem `search_offset` ignoriert wird
        und stattdessen ein Ereignis innerhalb des durchsuchbaren Bereichs gefunden wird.
        """
        data = np.zeros((self.data_len, 1))
        
        # Ein riesiges, unübersehbares Ereignis am Anfang der Daten (Index 10).
        # Dies liegt VOR dem search_offset von 72.
        data[10 : 10 + self.horizon] = 1000 
        
        # Ein kleineres, aber immer noch signifikantes Ereignis im durchsuchbaren Bereich.
        # Wir platzieren es bei Index 150.
        interesting_idx_in_search_area = 150
        data[interesting_idx_in_search_area + self.horizon : interesting_idx_in_search_area + 2 * self.horizon] = 50

        df = pd.DataFrame(data, columns=['A'])
        results = find_interesting_windows(df, self.horizon, self.seq_len)

        # Die Funktion sollte das Ereignis bei Index 150 finden, nicht das bei 10.
        self.assertEqual(results['A']['max_mean_diff_idx'], interesting_idx_in_search_area)

    def test_finds_correct_window_with_numpy_array(self):
        """
        Testet, ob die Funktion mit einem NumPy-Array funktioniert und ein klares
        Ereignis an einem bestimmten Index im Suchbereich findet.
        """
        # Wir verwenden np.zeros als Hintergrund, um Mehrdeutigkeiten durch zufälliges
        # Rauschen zu vermeiden. Dies stellt sicher, dass es nur zwei Fensterpaare
        # mit maximaler Varianzdifferenz gibt und argmax das erste findet.
        data = np.zeros((self.data_len, 2))
        
        # Erzeuge eine klare Varianz-Änderung für den zweiten Kanal bei Index 200
        target_index = 200
        # Stelle sicher, dass der Zielindex im durchsuchbaren Bereich liegt
        self.assertGreater(target_index, self.search_offset)
        
        # Das "Nachher"-Fenster wird eine hohe Varianz haben
        high_var_data = (np.array([1, -1] * (self.horizon // 2)) * 20)
        data[target_index + self.horizon : target_index + 2 * self.horizon, 1] = high_var_data

        # Die Funktion sollte Standard-Kanalnamen für NumPy-Arrays verwenden
        results = find_interesting_windows(data, self.horizon, self.seq_len)
        
        self.assertIn('channel_0', results)
        self.assertIn('channel_1', results)
        
        # Die Varianz-Änderung war in Kanal 1
        self.assertEqual(results['channel_1']['max_var_diff_idx'], target_index)

if __name__ == '__main__':
    unittest.main()