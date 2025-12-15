import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import re
import fastjet

class DataProcessor:
    """
    Handles loading and processing of HEP event data from Parquet files or in-memory dictionaries.
    """
    def __init__(self, path):
        """
        Initialize the DataProcessor.

        Args:
            path (str, Path, or dict): 
                - If str/Path: Path to the directory containing Parquet files.
                - If dict: In-memory dictionary of data (e.g. from Hugging Face).
        """
        self.files = {}
        self.memory_data = {}
        self.track_sources = {}
        self.event_index_map = {} # event_id -> {key: row_index}

        if isinstance(path, dict):
            # Check if this is Graph Mode data (edges/scores present)
            if 'edges' in path and 'scores' in path:
                self._init_from_graph(path)
            else:
                self._init_from_memory(path)
        else:
            self.path = Path(path)
            self.load_data()

    def _init_from_graph(self, data):
        """
        Initialize from raw graph connection data (GNN output).
        
        Expected keys in data:
            - hits: DataFrame OR List of DataFrames
            - edges: Tensor/Array OR List of arrays
            - scores: Tensor/Array OR List of arrays
            - truth: (Optional) Tensor/Array OR List of arrays
        """
        print("Initializing in Graph Mode...")
        self.files = {}
        self.memory_data = {}
        self.track_sources = {}
        self.event_index_map = {}
        self.graph_data = {} # event_id -> {edges, scores, truth}

        # Handle list vs single input
        raw_hits = data['hits']
        is_list = isinstance(raw_hits, list)
        
        # Normalize to lists
        list_hits = raw_hits if is_list else [raw_hits]
        list_edges = data['edges'] if is_list else [data['edges']]
        list_scores = data['scores'] if is_list else [data['scores']]
        
        raw_truth = data.get('truth', None)
        if raw_truth is None:
             list_truth = [None] * len(list_hits)
        else:
             list_truth = raw_truth if is_list else [raw_truth]

        all_tracker_hits = []
        all_particles = []

        for i, (hits_df, edges, scores, truth) in enumerate(zip(list_hits, list_edges, list_scores, list_truth)):
            # 1. Process Hits
            hits_df = hits_df.copy() # Enable modification
            if 'event_id' not in hits_df.columns:
                hits_df['event_id'] = i # Assign event ID based on list index
            
            # Ensure particle_id exists (Required for pipeline)
            if 'particle_id' not in hits_df.columns:
                hits_df['particle_id'] = 0
            
            # Append to master list
            hits_records = hits_df.to_dict('records')
            all_tracker_hits.extend(hits_records)
            
            # Particles (Auto-generate from hit IDs)
            unique_pids = hits_df['particle_id'].unique()
            particles_df = pd.DataFrame({'particle_id': unique_pids})
            particles_df['event_id'] = i
            particles_df['pdg_id'] = 0 
            
            if 'pT' in hits_df.columns:
                 max_pt = hits_df.groupby('particle_id')['pT'].max()
                 particles_df['pT'] = particles_df['particle_id'].map(max_pt)
            else:
                 particles_df['pT'] = 1.0
                 
            all_particles.extend(particles_df.to_dict('records'))

            # 2. Process Graph Data
            # Convert to numpy if tensor
            if hasattr(edges, 'cpu'): edges = edges.cpu().numpy()
            if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
            if truth is not None and hasattr(truth, 'cpu'): truth = truth.cpu().numpy()

            # Auto-generate Truth if missing
            if truth is None and 'particle_id' in hits_df.columns:
                print(f"Auto-generating truth labels for graph {i}...")
                src, dst = edges[0], edges[1]
                pids = hits_df['particle_id'].values
                # Safety check for indices
                if src.max() < len(pids) and dst.max() < len(pids):
                    pid_src = pids[src]
                    pid_dst = pids[dst]
                    truth = (pid_src == pid_dst).astype(int)
                else:
                    print(f"Warning: Edge indices out of bounds for event {i}. Cannot auto-generate truth.")
                    truth = np.array([]) # Empty truth if indices are bad

            self.graph_data[i] = {
                'edges': edges.tolist(), 
                'scores': scores.tolist(),
                'truth': truth.tolist() if truth is not None else []
            }

        self.memory_data['tracker_hits'] = all_tracker_hits
        if all_particles:
            self.memory_data['particles'] = all_particles
        
        # Call standard memory init
        self._init_from_memory(self.memory_data)

    def load_data(self):
        """
        Scan the directory and find the necessary Parquet files.
        Supports multiple files per category if filenames contain event ranges (e.g., 'events0-999').
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        if self.path.is_file():
            print(f"Warning: {self.path} is a file. Expecting a directory containing the dataset components.")
            return

        # Get all parquet files, excluding validation loss files
        all_parquet = list(self.path.rglob("*.parquet"))
        all_parquet = [p for p in all_parquet if "val_loss" not in p.name]

        # 1. Identify all 'keys' that start with 'tracks' present in the dataset
        potential_sources = set()
        for p in all_parquet:
             # Check parent directory name
             parent = p.parent.name
             if parent.startswith("tracks"):
                 potential_sources.add(parent)
             # Check filename if flat structure (less common for multi-source)
             if p.name.startswith("tracks") and "event" in p.name:
                  pass

        if not potential_sources:
             potential_sources.add('tracks')

        # Load non-track keys normally
        for key in ['particles', 'tracker_hits', 'calo_hits']:
            self._load_file_list(key, all_parquet)

        # Load track sources
        for source in potential_sources:
             self.track_sources[source] = []
             self._load_file_list(source, all_parquet, self.track_sources[source])

    def _load_file_list(self, key, all_parquet, target_list=None):
        """Helper to load files for a given key/source."""
        if target_list is None:
             self.files[key] = []
             target_list = self.files[key]

        # 0. Prioritize exact parent directory match (Best for multi-track folders)
        matches = [p for p in all_parquet if key == p.parent.name]

        # 1. If no parent match, try matching filename directly (Legacy/Flat structure)
        if not matches:
             matches = [p for p in all_parquet if key in p.name]
        
        # 2. If valid matches found (either way)
        if matches:
            print(f"Found {len(matches)} files for {key}")
            for p in matches:
                # Try to parse event range from filename: events(\d+)-(\d+)
                start_evt, end_evt = -1, -1
                
                # Pattern 1: Range (eventsX-Y)
                match_range = re.search(r'events(\d+)-(\d+)', p.name)
                if match_range:
                    start_evt = int(match_range.group(1))
                    end_evt = int(match_range.group(2))
                
                # Pattern 2: Single Event (event_N)
                match_single = re.search(r'event_(\d+)', p.name)
                if match_single and start_evt == -1:
                    start_evt = int(match_single.group(1))
                    end_evt = start_evt
                
                target_list.append({
                    "path": p,
                    "start": start_evt,
                    "end": end_evt
                })
            
            # Sort by start event to ensure logical order
            target_list.sort(key=lambda x: x['start'])
        else:
            if not key.startswith("tracks"): 
                print(f"Warning: Could not find file for {key}")

    def _init_from_memory(self, data: dict):
        """
        Initialize from in-memory dictionary (e.g. Hugging Face dataset).
        Builds an index map for fast random access to events.
        """
        # data is expected to be {'particles': dataset, 'tracks': dataset, ...}
        self.memory_data = data
        
        # Create an index map for fast access
        if 'particles' in self.memory_data:
            # 1. Identify Track Sources in Memory Data
            for key in self.memory_data.keys():
                if key.startswith("tracks"):
                    self.track_sources[key] = [] # Placeholder key existence matters
            
            # Ensure 'tracks' exists if no other sources found (or if it exists explicitly)
            if not self.track_sources and 'tracks' in self.memory_data:
                 self.track_sources['tracks'] = []

            print(f"Indexing in-memory data... Found track sources: {list(self.track_sources.keys())}")
            try:
                # Index particles
                particles = self.memory_data['particles']
                for i, row in enumerate(particles):
                    eid = row['event_id']
                    if eid not in self.event_index_map:
                        self.event_index_map[eid] = {}
                    if 'particles' not in self.event_index_map[eid]:
                        self.event_index_map[eid]['particles'] = []
                    self.event_index_map[eid]['particles'].append(i)
                
                # Index other keys (including all track sources)
                keys_to_index = ['tracker_hits', 'calo_hits'] + list(self.track_sources.keys())
                
                for key in keys_to_index:
                    if key in self.memory_data:
                        for i, row in enumerate(self.memory_data[key]):
                            eid = row['event_id']
                            if eid not in self.event_index_map:
                                self.event_index_map[eid] = {}
                            if key not in self.event_index_map[eid]:
                                self.event_index_map[eid][key] = []
                            self.event_index_map[eid][key].append(i)
                
                print(f"Indexing Complete. Map keys: {list(self.event_index_map.keys())}")
            except Exception as e:
                print(f"Error indexing memory data: {e}")

    def _load_parquet_event(self, path, event_id):
        """
        Load data for a specific event from a Parquet file using filters.
        """
        try:
            # Use filters to load only the specific event
            df = pd.read_parquet(path, filters=[('event_id', '==', event_id)])
            
            # Explode common columns if they exist (handling variable length data)
            df = df.explode([col for col in df.columns if col != 'event_id'])
            
            if 'hit_ids' in df.columns:
                 df = df.explode('hit_ids')
            
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading event {event_id} from {path}: {e}")
            return pd.DataFrame()

    def get_event_list(self):
        """
        Return a dictionary containing a list of available event IDs.
        """
        # Helper to get event IDs from a list of files
        def get_ids_from_files(file_list):
             ids = set()
             for file_info in file_list:
                if file_info['start'] != -1 and file_info['end'] != -1:
                    ids.update(range(file_info['start'], file_info['end'] + 1))
                else:
                    try:
                        df = pd.read_parquet(file_info['path'], columns=['event_id'])
                        ids.update(df['event_id'].unique().tolist())
                    except: 
                        pass
             return ids

        if self.memory_data:
            events = sorted(list(self.event_index_map.keys()))
            sources = list(self.track_sources.keys())
            if not sources: sources = []
            else:
                 if 'tracks' in sources:
                      sources.remove('tracks')
                      sources.insert(0, 'tracks')
            return {"events": events, "track_sources": sources}

        if 'particles' in self.files and self.files['particles']:
            all_events = get_ids_from_files(self.files['particles'])
            
            # Available track sources
            sources = list(self.track_sources.keys())
            if not sources:
                 sources = []
            else:
                 # Ensure 'tracks' is first if exists
                 if 'tracks' in sources:
                      sources.remove('tracks')
                      sources.insert(0, 'tracks')
            
            return {"events": sorted(list(all_events)), "track_sources": sources}
        return {"events": [], "track_sources": []}

    def process_event(self, event_id: str, track_source: str = "tracks"):
        """
        Load and process data for a specific event ID.

        Args:
            event_id (str): The event ID.
            track_source (str): The specific track source key to load (e.g. 'tracks', 'tracks_algo').
                                Defaults to 'tracks'.

        Returns:
            dict: A dictionary with 'tracks', 'calo_hits', 'all_tracker_hits', and 'metadata'.
        """
        try:
            n = int(event_id)
        except ValueError:
            raise ValueError(f"Invalid event ID: {event_id}")

        if self.memory_data:
            # --- Load from Memory ---
            if n not in self.event_index_map:
                return {"error": f"Event {n} not found in memory data"}
            
            idx_map = self.event_index_map[n]
            
            # Particles
            if 'particles' not in idx_map:
                return {"error": "No particle data for this event"}
            
            p_idxs = idx_map['particles']
            # Ensure list
            if not isinstance(p_idxs, list): p_idxs = [p_idxs]
            p_data = [self.memory_data['particles'][i] for i in p_idxs]
            particles = pd.DataFrame(p_data)
            
            # Tracker Hits
            tracker_hits = pd.DataFrame()
            if 'tracker_hits' in idx_map:
                th_idxs = idx_map['tracker_hits']
                if not isinstance(th_idxs, list): th_idxs = [th_idxs]
                th_data = [self.memory_data['tracker_hits'][i] for i in th_idxs]
                tracker_hits = pd.DataFrame(th_data)
                
            # Calo Hits
            calo_hits = pd.DataFrame()
            if 'calo_hits' in idx_map:
                ch_idxs = idx_map['calo_hits']
                if not isinstance(ch_idxs, list): ch_idxs = [ch_idxs]
                ch_data = [self.memory_data['calo_hits'][i] for i in ch_idxs]
                calo_hits = pd.DataFrame(ch_data)

            # Reco Tracks
            tracks_df = pd.DataFrame()
            if track_source in idx_map:
                t_idxs = idx_map[track_source]
                if not isinstance(t_idxs, list): t_idxs = [t_idxs]
                t_data = [self.memory_data[track_source][i] for i in t_idxs]
                tracks_df = pd.DataFrame(t_data)


        else:
            # --- Load from Files ---
            if 'particles' not in self.files or not self.files['particles']:
                return {"error": "No particle data found"}

            # Helper to find correct file based on event ID
            def get_file_for_event(source_list, event_id):
                if not source_list:
                    return None
                
                # 1. Check ranges
                for f in source_list:
                    if f['start'] <= event_id <= f['end']:
                        return f['path']
                
                # 2. Fallback
                for f in source_list:
                    if f['start'] == -1:
                         if len(source_list) == 1:
                             return f['path']
                return None

            p_path = get_file_for_event(self.files.get('particles'), n)
            if not p_path:
                 return {"error": f"Event {n} not found in particle files"}

            # Load data for this event ONLY
            particles = self._load_parquet_event(p_path, n)
            
            tracker_hits = pd.DataFrame()
            th_path = get_file_for_event(self.files.get('tracker_hits'), n)
            if th_path:
                tracker_hits = self._load_parquet_event(th_path, n)

            calo_hits = pd.DataFrame()
            ch_path = get_file_for_event(self.files.get('calo_hits'), n)
            if ch_path:
                calo_hits = self._load_parquet_event(ch_path, n)

            tracks_df = pd.DataFrame()
            # Use specific track source
            current_track_list = self.track_sources.get(track_source)
            # Fallback if specific source not found (to avoid crash), though UI should prevent this
            if current_track_list is None and 'tracks' in self.track_sources:
                 current_track_list = self.track_sources['tracks']
            
            t_path = get_file_for_event(current_track_list, n)
            if t_path:
                tracks_df = self._load_parquet_event(t_path, n)

        # --- Common Processing ---
        
        # 1. Pre-process Tracker Hits
        if not tracker_hits.empty:
             if 'HIT_ID' not in tracker_hits.columns:
                tracker_hits = tracker_hits.reset_index(drop=True)
                tracker_hits = tracker_hits.reset_index().rename(columns={'index': 'HIT_ID'})
             tracker_hits = tracker_hits.apply(pd.to_numeric, errors='coerce')

        # 2. Pre-process Calo Hits
        if not calo_hits.empty:
             if 'HIT_ID' not in calo_hits.columns:
                calo_hits = calo_hits.reset_index(drop=True)
                calo_hits = calo_hits.reset_index().rename(columns={'index': 'HIT_ID'})
             
             # Explode list columns (e.g. contrib_particle_ids)
             cols_to_explode = ['contrib_particle_ids', 'contrib_energies', 'contrib_times']
             
             # Check which columns are actually lists and present
             actual_explode_cols = []
             for col in cols_to_explode:
                 if col in calo_hits.columns:
                     sample = calo_hits[col].dropna().iloc[0] if not calo_hits[col].dropna().empty else None
                     if isinstance(sample, (list, np.ndarray)):
                         actual_explode_cols.append(col)
             
             if actual_explode_cols:
                 try:
                     # Explode all list columns simultaneously
                     calo_hits = calo_hits.explode(actual_explode_cols)
                 except ValueError as e:
                     print(f"Warning: Simultaneous explode failed ({e}). Falling back to sequential.")
                     for col in actual_explode_cols:
                         calo_hits = calo_hits.explode(col)

             all_numeric_cols = ['cell_id', 'total_energy', 'x', 'y', 'z','contrib_particle_ids', 'contrib_energies', 'contrib_times']
             for col in all_numeric_cols:
                 if col in calo_hits.columns:
                     calo_hits[col] = pd.to_numeric(calo_hits[col], errors='coerce')

        # 3. Apply static cuts (Vertex position)
        if 'vx' in particles.columns and 'vy' in particles.columns:
            particles = particles[abs(particles['vx']) < 1]
            particles = particles[abs(particles['vy']) < 1]
        
        # 4. Calculate pT and PDG ID
        if 'px' in particles.columns and 'py' in particles.columns:
            particles['px'] = pd.to_numeric(particles['px'], errors='coerce')
            particles['py'] = pd.to_numeric(particles['py'], errors='coerce')
            particles['pT'] = np.sqrt(particles['px']**2 + particles['py']**2)
        else:
             if 'pT' not in particles.columns:
                 particles['pT'] = 0.0
        
        # Get particle IDs to match
        particles_id = particles["particle_id"].unique()
        
        # Create lookup maps for fast enrichment
        pt_map = particles.set_index('particle_id')['pT'].to_dict()
        pdg_map = particles.set_index('particle_id')['pdg_id'].to_dict()

        # 5. Process ALL Tracker Hits (for point cloud)
        all_event_tracker_hits = []
        has_volumes = False
        
        if not tracker_hits.empty:
            filtered_tracker_hits = tracker_hits[tracker_hits["particle_id"].isin(particles_id)].copy()
            
            filtered_tracker_hits['pT'] = filtered_tracker_hits['particle_id'].map(pt_map)
            filtered_tracker_hits['pdg_id'] = filtered_tracker_hits['particle_id'].map(pdg_map)
            filtered_tracker_hits = filtered_tracker_hits.dropna(subset=['pT', 'pdg_id'])
            
            # Check for volume_id
            cols_to_keep = ['x', 'y', 'z', 'pT', 'pdg_id', 'particle_id']
            if 'HIT_ID' in filtered_tracker_hits.columns:
                cols_to_keep.append('HIT_ID')
            elif 'hit_id' in filtered_tracker_hits.columns:
                cols_to_keep.append('hit_id')

            if 'volume_id' in filtered_tracker_hits.columns:
                cols_to_keep.append('volume_id')
                has_volumes = True
                filtered_tracker_hits['volume_id'] = pd.to_numeric(filtered_tracker_hits['volume_id'], errors='coerce').fillna(-1).astype(int)
            
            all_event_tracker_hits = filtered_tracker_hits[cols_to_keep].to_dict(orient='records')

        # 6. Process Tracks (grouped hits)
        event_tracks = []
        if not tracker_hits.empty:
            tracker_hits_for_tracks = tracker_hits[tracker_hits["particle_id"].isin(particles_id)].copy()
            tracker_hits_for_tracks['r'] = np.sqrt(tracker_hits_for_tracks['x']**2 + tracker_hits_for_tracks['y']**2 + tracker_hits_for_tracks['z']**2)
            tracker_hits_for_tracks = tracker_hits_for_tracks.sort_values(by=['particle_id', 'r'])

            for particle_id, group_of_hits in tracker_hits_for_tracks.groupby('particle_id'):
                cols = ['x', 'y', 'z']
                if 'volume_id' in group_of_hits.columns:
                    cols.append('volume_id')
                    group_of_hits['volume_id'] = pd.to_numeric(group_of_hits['volume_id'], errors='coerce').fillna(-1).astype(int)
                
                points = group_of_hits[cols].to_dict(orient='records')
                
                if len(points) > 1:
                    # Collect hit_ids for graph generation
                    hit_ids_list = []
                    if 'hit_id' in group_of_hits.columns:
                        hit_ids_list = group_of_hits['hit_id'].tolist()
                    elif 'HIT_ID' in group_of_hits.columns:
                        hit_ids_list = group_of_hits['HIT_ID'].tolist()

                    # A. Find Matching Reco Track
                    track_info = {
                        'd0': 0.0, 'z0': 0.0, 'phi': 0.0, 'theta': 0.0, 'qop': 0.0,
                        'has_reco': False
                    }
                    
                    if not tracks_df.empty and 'majority_particle_id' in tracks_df.columns:
                        # Ensure robust type matching (int vs float) by converting both to int
                        pid_target = int(particle_id)
                        
                        # Handle potential NaN or float types in tracks_df
                        try:
                            # Create a temporary safe comparison column
                            tracks_df['temp_maj_pid'] = pd.to_numeric(tracks_df['majority_particle_id'], errors='coerce').fillna(-1).astype(int)
                            reco_track = tracks_df[tracks_df['temp_maj_pid'] == pid_target]
                            
                            if not reco_track.empty:
                                row = reco_track.iloc[0]
                                track_info = {
                                    'd0': float(row['d0']) if 'd0' in row else 0.0,
                                    'z0': float(row['z0']) if 'z0' in row else 0.0,
                                    'phi': float(row['phi']) if 'phi' in row else 0.0,
                                    'theta': float(row['theta']) if 'theta' in row else 0.0,
                                    'qop': float(row['qop']) if 'qop' in row else 0.0,
                                    'has_reco': True
                                }
                        except Exception as e:
                            print(f"Error matching tracks: {e}")

                    # B. Get Associated Calo Contributions
                    calo_data = []
                    if not calo_hits.empty and 'contrib_particle_ids' in calo_hits.columns:
                         # Ensure robust type matching for calo hits too
                        try:
                            pid_target = int(particle_id)
                            # Assuming contrib_particle_ids was already exploded and cast, but let's be safe
                            # (It was cast to numeric above, but float sensitivity can be an issue)
                            associated_calo = calo_hits[np.isclose(calo_hits['contrib_particle_ids'], pid_target, atol=0.1)] 
                            
                            for _, hit in associated_calo.iterrows():
                                calo_data.append({
                                    'energy': float(hit['contrib_energies']) if 'contrib_energies' in hit else 0.0,
                                    'detector': str(hit['detector']) if 'detector' in hit else 'Unknown',
                                    'cell_id': str(hit['cell_id']) if 'cell_id' in hit else ''
                                })
                        except Exception as e:
                             pass

                    event_tracks.append({
                        'particle_id': int(particle_id),
                        'pT': float(pt_map.get(particle_id, 0)),
                        'pdg_id': int(pdg_map.get(particle_id, 0)),
                        'hit_ids': hit_ids_list,
                        'reco_info': track_info,
                        'calo_hits': calo_data,
                        'points': points
                    })

        # Calculate Primary Vertex (Mean of particle production vertices)
        primary_vertex = [0.0, 0.0, 0.0]
        if not particles.empty:
            v_cols = [c for c in ['vx', 'vy', 'vz'] if c in particles.columns]
            if v_cols:
                 # Take mean of available vertex columns
                 means = particles[v_cols].mean()
                 primary_vertex = [
                     float(means.get('vx', 0.0)),
                     float(means.get('vy', 0.0)),
                     float(means.get('vz', 0.0))
                 ]

        # 7. Process Calo Hits
        event_calo_hits = []
        if not calo_hits.empty and 'contrib_particle_ids' in calo_hits.columns:
            # Note: calo_hits is already exploded above
            
            calo_hits['contrib_particle_ids'] = pd.to_numeric(calo_hits['contrib_particle_ids'], errors='coerce')
            calo_hits = calo_hits.dropna(subset=['contrib_particle_ids'])
            calo_hits = calo_hits[calo_hits['contrib_particle_ids'].isin(particles_id)]
            
            calo_hits['pT'] = calo_hits['contrib_particle_ids'].map(pt_map)
            calo_hits['pdg_id'] = calo_hits['contrib_particle_ids'].map(pdg_map)
            calo_hits = calo_hits.dropna(subset=['pT', 'pdg_id'])
            
            calo_hits['contrib_energies'] = pd.to_numeric(calo_hits['contrib_energies'], errors='coerce')

            temp_calo_hits = calo_hits[['x', 'y', 'z', 'contrib_energies', 'pT', 'pdg_id', 'contrib_particle_ids']].to_dict(orient='records')
            
            for hit in temp_calo_hits:
                event_calo_hits.append({
                    'x': hit['x'], 'y': hit['y'], 'z': hit['z'],
                    'energy': hit['contrib_energies'],
                    'pT': hit['pT'],
                    'pdg_id': hit['pdg_id'],
                    'particle_id': hit['contrib_particle_ids']
                })

        # 8. Graph Data (If available)
        graph_response = {}
        if hasattr(self, 'graph_data') and n in self.graph_data:
             graph_response = self.graph_data[n]

        # 9. Jet Clustering
        jet_data = {"truth": [], "reco": []}
        if fastjet:
            # --- Truth Jets (from particles) ---
            truth_vectors = []
            if not particles.empty:
                # Ensure we have momentum columns
                if {'px', 'py', 'pz'}.issubset(particles.columns):
                    for _, p in particles.iterrows():
                        px, py, pz = p['px'], p['py'], p['pz']
                        # Calculate energy if missing (assume massless if not provided)
                        e = p['energy'] if 'energy' in p else np.sqrt(px**2 + py**2 + pz**2)
                        truth_vectors.append(fastjet.PseudoJet(px, py, pz, e))
                elif {'pT', 'phi', 'eta'}.issubset(particles.columns):
                     for _, p in particles.iterrows():
                        pt, phi, eta = p['pT'], p['phi'], p['eta']
                        e = p['energy'] if 'energy' in p else pt * np.cosh(eta) # Massless approx
                        vec = fastjet.PseudoJet()
                        vec.reset_PtYPhiM(pt, eta, phi, 0.0) # Massless
                        truth_vectors.append(vec)

            if truth_vectors:
                jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
                cluster_seq = fastjet.ClusterSequence(truth_vectors, jet_def)
                inclusive_jets = cluster_seq.inclusive_jets(5.0)
                
                for j in inclusive_jets:
                     jet_data["truth"].append({
                         "pt": j.pt(),
                         "eta": j.eta(),
                         "phi": j.phi(),
                         "energy": j.e(),
                         "n_constituents": len(j.constituents())
                     })

            # --- Reco Jets (Tracks + Calo) ---
            reco_vectors = []
            reco_source_objects = [] # Parallel list to store source data

            # 1. From Reco Tracks
            for t in event_tracks:
                if t['reco_info'].get('has_reco'):
                     info = t['reco_info']
                     qop = info.get('qop', 0.0)
                     if abs(qop) > 1e-9:
                         p = abs(1.0 / qop)
                         theta = info.get('theta', 0.0)
                         phi = info.get('phi', 0.0)
                         
                         pt = p * np.sin(theta)
                         pz = p * np.cos(theta)
                         px = pt * np.cos(phi)
                         py = pt * np.sin(phi)
                         e = np.sqrt(p**2 + 0.13957**2) 
                         
                         pj = fastjet.PseudoJet(px, py, pz, e)
                         pj.set_user_index(len(reco_source_objects))
                         reco_vectors.append(pj)
                         reco_source_objects.append({'type': 'track', 'data': t})

            # 2. From Calo Hits
            for c in event_calo_hits:
                e = c['energy']
                if e > 0.1: 
                    dist = np.sqrt(c['x']**2 + c['y']**2 + c['z']**2)
                    if dist > 0:
                        px = (c['x'] / dist) * e
                        py = (c['y'] / dist) * e
                        pz = (c['z'] / dist) * e
                        pj = fastjet.PseudoJet(px, py, pz, e)
                        pj.set_user_index(len(reco_source_objects))
                        reco_vectors.append(pj)
                        reco_source_objects.append({'type': 'calo', 'data': c})
            
            if reco_vectors:
                 jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
                 cluster_seq = fastjet.ClusterSequence(reco_vectors, jet_def)
                 inclusive_jets = cluster_seq.inclusive_jets(5.0)
                 
                 for j in inclusive_jets:
                     # Calculate max extent
                     max_extent = 0.0
                     for c in j.constituents():
                         idx = c.user_index()
                         if idx >= 0 and idx < len(reco_source_objects):
                             obj = reco_source_objects[idx]
                             if obj['type'] == 'track':
                                 # Find max R of track points
                                 for p in obj['data']['points']:
                                     r = np.sqrt(p['x']**2 + p['y']**2 + p['z']**2)
                                     if r > max_extent: max_extent = r
                             elif obj['type'] == 'calo':
                                 d = obj['data']
                                 r = np.sqrt(d['x']**2 + d['y']**2 + d['z']**2)
                                 if r > max_extent: max_extent = r
                     
                     if max_extent == 0.0: max_extent = 1000.0 # Default fallback if no hits?

                     jet_data["reco"].append({
                         "pt": j.pt(),
                         "eta": j.eta(),
                         "phi": j.phi(),
                         "energy": j.e(),
                         "n_constituents": len(j.constituents()),
                         "max_extent": max_extent
                     })

        return {
            "metadata": {
                "event_id": int(n),
                "has_reco": bool(event_tracks),
                "has_calo": bool(event_calo_hits),
                "has_pdg": True, # Always true if particles exist
                "has_volumes": has_volumes,
                "has_graph": bool(graph_response),
                "has_jets": bool(jet_data["truth"] or jet_data["reco"])
            },
            "jets": jet_data,
            "vertex": primary_vertex,
            "tracks": event_tracks,
            "calo_hits": event_calo_hits,
            "all_tracker_hits": all_event_tracker_hits,
            "graph_data": graph_response
        }
