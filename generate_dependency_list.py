import ast
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

def find_imports(file_path: Path, project_root: Path) -> set:
    """
    Analysiert eine einzelne Python-Datei und findet alle lokalen Projekt-Importe.
    Gibt ein Set von aufgelösten Pfaden zu den importierten Dateien zurück.
    """
    if not file_path.exists():
        return set()

    # Verzeichnisse, die Python-Code enthalten können, für die Auflösung
    source_dirs = {p.parent for p in project_root.rglob('*.py')}
    source_dirs.add(project_root)

    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Warnung: Konnte Datei nicht parsen oder lesen: {file_path}. Fehler: {e}")
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # Relative Imports (z.B. `from . import utils` oder `from ..models import core`)
            if node.level > 0:
                # Baue den Basispfad für den relativen Import
                base_path = file_path.parent
                for _ in range(node.level - 1):
                    base_path = base_path.parent
                
                module_path_part = node.module.replace('.', '/') if node.module else ''
                potential_path = (base_path / module_path_part).resolve()
            # Absolute Projekt-Imports (z.B. `from utils import helpers`)
            else:
                module_path_part = node.module.replace('.', '/')
                potential_path = (project_root / module_path_part).resolve()
            
            # Überprüfe, ob es sich um eine .py-Datei oder ein Paket handelt
            if potential_path.is_file() and potential_path.with_suffix('.py').exists():
                 imports.add(potential_path.with_suffix('.py'))
            elif (potential_path / '__init__.py').exists():
                imports.add(potential_path / '__init__.py')
            elif potential_path.with_suffix('.py').exists():
                 imports.add(potential_path.with_suffix('.py'))


        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_path_part = alias.name.replace('.', '/')
                # Suche in allen bekannten Quellcode-Verzeichnissen
                for src_dir in source_dirs:
                    potential_path = (src_dir / module_path_part).resolve()
                    
                    if potential_path.with_suffix('.py').exists() and potential_path.with_suffix('.py').is_file():
                         if project_root in potential_path.parents:
                            imports.add(potential_path.with_suffix('.py'))
                         break # Nimm den ersten Treffer
                    elif (potential_path / '__init__.py').exists():
                        if project_root in potential_path.parents:
                            imports.add(potential_path / '__init__.py')
                        break # Nimm den ersten Treffer
    
    # Filtere alle Pfade heraus, die nicht innerhalb des Projektverzeichnisses liegen
    return {imp for imp in imports if project_root in imp.parents or imp.parent == project_root}


def create_recursive_dependency_graph(entry_points: list, project_root_str: str):
    """
    Erstellt einen Graphen der rekursiven Abhängigkeiten, inklusive der Beziehungen zwischen den Dateien.
    Verwendet eine Breitensuche (BFS), um alle Abhängigkeiten von den Einstiegspunkten aus zu finden.
    """
    project_root = Path(project_root_str).resolve()
    graph = nx.DiGraph()
    
    # Warteschlange für zu besuchende Dateien (BFS)
    queue = []
    # Set, um bereits besuchte oder in die Warteschlange gestellte Dateien zu verfolgen,
    # um Zyklen und redundante Arbeit zu vermeiden.
    visited = set()

    # Initialisiere die Warteschlange mit allen Einstiegspunkten
    for entry_point_str in entry_points:
        entry_point = (project_root / entry_point_str).resolve()
        if entry_point.exists() and entry_point not in visited:
            queue.append(entry_point)
            visited.add(entry_point)
        elif not entry_point.exists():
            print(f"Warnung: Einstiegspunkt '{entry_point}' nicht gefunden. Überspringe.")

    while queue:
        current_file = queue.pop(0)
        # Relative Pfade für sauberere Knotenbeschriftungen im Graphen verwenden
        importer_node = str(current_file.relative_to(project_root))

        try:
            # Finde alle direkten lokalen Importe für die aktuelle Datei
            imported_files = find_imports(current_file, project_root)
        except Exception as e:
            print(f"Warnung: Datei '{current_file}' konnte nicht verarbeitet werden: {e}")
            continue

        for imported_file in imported_files:
            imported_node = str(imported_file.relative_to(project_root))
            
            # Füge eine Kante von der aktuellen Datei zu ihrer Abhängigkeit hinzu.
            # networkx erstellt Knoten automatisch, wenn sie nicht existieren.
            graph.add_edge(importer_node, imported_node)
            
            # Wenn wir eine neue, noch nicht besuchte Datei gefunden haben, fügen wir sie
            # zur Warteschlange hinzu, um ihre Abhängigkeiten ebenfalls zu prüfen.
            if imported_file not in visited:
                visited.add(imported_file)
                queue.append(imported_file)
                
    return graph

if __name__ == "__main__":
    import json
    # Das Projektverzeichnis wird dynamisch als das Verzeichnis bestimmt, in dem dieses Skript liegt.
    PROJECT_ROOT = Path(__file__).resolve().parent
    
    # Liste der Haupt-Einstiegspunkte für Training/Optimierung, basierend auf der Projektstruktur.
    core_entry_points = [
        "scripts/run_benchmark.py",
        "optuna_full_search.py",
        "optimize.py",
    ]
    
    # Liste der Skripte für Inferenz und Analyse.
    utility_scripts = [
        "simple_inference.py",
        "inference_and_plot.py",
        "analyze_predictions.py",
    ]
    
    print("===================================================================")
    print("Analysiere Abhängigkeiten...")
    print(f"Projektverzeichnis: {PROJECT_ROOT}")
    print("===================================================================")

    # Erstelle den rekursiven Abhängigkeitsgraphen
    entry_points_for_graph = core_entry_points + utility_scripts
    recursive_graph = create_recursive_dependency_graph(entry_points_for_graph, PROJECT_ROOT)
    
    # Konvertiere den Graphen in ein Dictionary für die Ausgabe
    graph_data = {
        node: list(recursive_graph.neighbors(node))
        for node in recursive_graph.nodes()
    }
    
    # Alle benötigten Dateien sind die Knoten im Graphen
    all_required_files = set(recursive_graph.nodes())
    
    print(f"\nInsgesamt {len(all_required_files)} benötigte Python-Skripte gefunden:")
    # Konvertiere das Set in eine sortierte Liste für eine saubere Ausgabe + Graph Ausgabe
    final_file_list = sorted(list(all_required_files))
    for file_path in final_file_list:
        print(f"- {file_path}")

    # Zeichne den Graphen (verbessert)
    print("\nErstelle Abhängigkeitsgraph...")
    plt.figure(figsize=(55, 55)) 
    # Ein Layout, das oft gut für gerichtete Graphen funktioniert
    try:
        pos = nx.nx_agraph.graphviz_layout(recursive_graph, prog='dot')
    except:
        print("Warnung: 'pygraphviz' nicht gefunden. Verwende 'spring_layout'. Für eine bessere Darstellung 'pip install pygraphviz' installieren.")
        pos = nx.spring_layout(recursive_graph, k=0.2, iterations=30, seed=42)

    nx.draw(
        recursive_graph,
        pos,
        with_labels=True,
        labels={node: node.replace('.py', '').replace('/', '.').strip('.') for node in recursive_graph.nodes()},  # Clean labels
        node_size=2000,
        node_color="#a8dacc",
        node_shape="s", # Quadratische Knoten
        font_size=8,
        font_weight="normal",
        font_color="#1d3557",
        edge_color="#e63946",
        arrowsize=15,
        width=1.5,
    )
    plt.title("Rekursiver Abhängigkeitsgraph der Python-Dateien", size=20)
    plt.tight_layout()
    plt.savefig("recursive_dependency_graph.png", dpi=400)
    
    # Ausgabe des Graphen als JSON
    with open("dependency_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)

    print("\n===================================================================")
    print("NICHT VERGESSEN: Die `requirements.txt` und relevante Konfigurationsdateien (.json, .sh) müssen ebenfalls kopiert werden.")
    print("===================================================================")
    print("\nRekursiver Abhängigkeitsgraph als 'recursive_dependency_graph.png' gespeichert.")
