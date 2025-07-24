import shutil
from pathlib import Path

# Das Stammverzeichnis des Projekts (dort, wo das Skript liegt).
project_root = Path(__file__).parent

print(f"Starte Cache-Bereinigung in: {project_root}\n")

# --- Muster für Cache-Dateien und -Ordner, die gelöscht werden sollen ---
patterns_to_delete = [
    '__pycache__',
    '*.pyc'
]

# 1. BEREINIGUNGSPHASE
# ======================
print("--- Phase 1: Suche und lösche Elemente ---")
found_items_count = 0
for pattern in patterns_to_delete:
    # .rglob() findet alle passenden Dateien/Ordner im gesamten Verzeichnisbaum
    for path in project_root.rglob(pattern):
        found_items_count += 1
        try:
            if path.is_dir():
                print(f"Lösche Verzeichnis: {path}")
                shutil.rmtree(path)
            elif path.is_file():
                print(f"Lösche Datei: {path}")
                path.unlink()
        except OSError as e:
            print(f"🚨 FEHLER beim Löschen von {path}: {e.strerror}")

if found_items_count == 0:
    print("Keine Cache-Dateien oder -Verzeichnisse gefunden.")
else:
    print(f"\n{found_items_count} Element(e) zum Löschen gefunden und verarbeitet.")


# 2. FINALE TESTPHASE
# =====================
print("\n--- Phase 2: Abschließender Test, ob noch Reste vorhanden sind ---")

remaining_items = []
for pattern in patterns_to_delete:
    # Wir führen exakt dieselbe Suche erneut durch
    remaining_items.extend(project_root.rglob(pattern))

if not remaining_items:
    print("✅ Test erfolgreich: Keine Cache-Reste mehr gefunden.")
else:
    print("🚨 ACHTUNG: Nach der Bereinigung wurden noch folgende Reste gefunden:")
    for path in remaining_items:
        print(f"  -> {path}")
    print("\nBitte überprüfe mögliche Probleme wie Dateiberechtigungen.")

print("\nBereinigung vollständig abgeschlossen.")