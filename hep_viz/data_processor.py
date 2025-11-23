import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import re

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
        self.event_index_map = {} # event_id -> {key: row_index}

        if isinstance(path, dict):
            self._init_from_memory(path)
        else:
            self.path = Path(path)
            self.files = {} # key -> list of {path, start, end}
            self.load_data()

    def load_data(self):
        """
        Scan the directory and find the necessary Parquet files.
        Supports multiple files per category if filenames contain event ranges (e.g., 'events0-999').
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        # Define patterns to search for
        keys = ['particles', 'tracks', 'tracker_hits', 'calo_hits']
        
        if self.path.is_file():
            print(f"Warning: {self.path} is a file. Expecting a directory containing the dataset components.")
            return

        # Get all parquet files, excluding validation loss files
        all_parquet = list(self.path.rglob("*.parquet"))
        all_parquet = [p for p in all_parquet if "val_loss" not in p.name]

        for key in keys:
            self.files[key] = []
            # 1. Try matching filename directly
            matches = [p for p in all_parquet if key in p.name]
            
            # 2. If no filename match, try matching parent directory name
            if not matches:
                matches = [p for p in all_parquet if key in p.parent.name]
            
            if matches:
                print(f"Found {len(matches)} files for {key}")
                for p in matches:
                    # Try to parse event range from filename: events(\d+)-(\d+)
                    # Example: hard_scatter.ttbar.v1.truth.particles.events0-9.parquet
                    match = re.search(r'events(\d+)-(\d+)', p.name)
                    start_evt, end_evt = -1, -1
                    if match:
                        start_evt = int(match.group(1))
                        end_evt = int(match.group(2))
                    
                    self.files[key].append({
                        "path": p,
                        "start": start_evt,
                        "end": end_evt
                    })
                
                # Sort by start event to ensure logical order
                self.files[key].sort(key=lambda x: x['start'])
            else:
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
            print("Indexing in-memory data...")
            try:
                # Index particles
                particles = self.memory_data['particles']
                for i, row in enumerate(particles):
                    eid = row['event_id']
                    if eid not in self.event_index_map:
                        self.event_index_map[eid] = {}
                    self.event_index_map[eid]['particles'] = i
                
                # Index other keys
                for key in ['tracks', 'tracker_hits', 'calo_hits']:
                    if key in self.memory_data:
                        for i, row in enumerate(self.memory_data[key]):
                            eid = row['event_id']
                            if eid not in self.event_index_map:
                                self.event_index_map[eid] = {}
                            self.event_index_map[eid][key] = i
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
        if self.memory_data:
            events = sorted(list(self.event_index_map.keys()))
            return {"events": events}

        if 'particles' in self.files and self.files['particles']:
            all_events = set()
            try:
                for file_info in self.files['particles']:
                    # Optimization: if range is known from filename, just add the range
                    if file_info['start'] != -1 and file_info['end'] != -1:
                        # ranges in filenames are usually inclusive
                        all_events.update(range(file_info['start'], file_info['end'] + 1))
                    else:
                        # Fallback: read file to find event IDs
                        df = pd.read_parquet(file_info['path'], columns=['event_id'])
                        all_events.update(df['event_id'].unique().tolist())
                
                return {"events": sorted(list(all_events))}
            except Exception as e:
                print(f"Error reading event list: {e}")
                return {"events": []}
        return {"events": []}

    def process_event(self, event_id: str):
        """
        Load and process data for a specific event ID.
        Returns a dictionary with 'tracks', 'calo_hits', 'all_tracker_hits', and 'metadata'.
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
            
            p_idx = idx_map['particles']
            p_data = self.memory_data['particles'][p_idx]
            particles = pd.DataFrame(p_data)
            
            # Tracker Hits
            tracker_hits = pd.DataFrame()
            if 'tracker_hits' in idx_map:
                th_idx = idx_map['tracker_hits']
                th_data = self.memory_data['tracker_hits'][th_idx]
                tracker_hits = pd.DataFrame(th_data)
                
            # Calo Hits
            calo_hits = pd.DataFrame()
            if 'calo_hits' in idx_map:
                ch_idx = idx_map['calo_hits']
                ch_data = self.memory_data['calo_hits'][ch_idx]
                calo_hits = pd.DataFrame(ch_data)

        else:
            # --- Load from Files ---
            if 'particles' not in self.files or not self.files['particles']:
                return {"error": "No particle data found"}

            # Helper to find correct file based on event ID
            def get_file_for_event(key, event_id):
                if key not in self.files:
                    return None
                
                # 1. Check ranges
                for f in self.files[key]:
                    if f['start'] <= event_id <= f['end']:
                        return f['path']
                
                # 2. Fallback: if ranges are unknown (-1), check all files (slow)
                # Ideally, we should avoid this by ensuring filenames have ranges.
                for f in self.files[key]:
                    if f['start'] == -1:
                         if len(self.files[key]) == 1:
                             return f['path']
                return None

            p_path = get_file_for_event('particles', n)
            if not p_path:
                 return {"error": f"Event {n} not found in particle files"}

            # Load data for this event ONLY
            particles = self._load_parquet_event(p_path, n)
            
            tracker_hits = pd.DataFrame()
            th_path = get_file_for_event('tracker_hits', n)
            if th_path:
                tracker_hits = self._load_parquet_event(th_path, n)

            calo_hits = pd.DataFrame()
            ch_path = get_file_for_event('calo_hits', n)
            if ch_path:
                calo_hits = self._load_parquet_event(ch_path, n)

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
             # This is necessary because a single calo hit can have contributions from multiple particles.
             cols_to_explode = ['contrib_particle_ids', 'contrib_energies', 'contrib_times']
             
             # Check which columns are actually lists and present
             actual_explode_cols = []
             for col in cols_to_explode:
                 if col in calo_hits.columns:
                     # Check first non-null value to see if it's a list
                     sample = calo_hits[col].dropna().iloc[0] if not calo_hits[col].dropna().empty else None
                     if isinstance(sample, (list, np.ndarray)):
                         actual_explode_cols.append(col)
             
             if actual_explode_cols:
                 try:
                     # Explode all list columns simultaneously to avoid Cartesian product
                     calo_hits = calo_hits.explode(actual_explode_cols)
                 except ValueError as e:
                     # Fallback for older pandas or mismatched lengths
                     print(f"Warning: Simultaneous explode failed ({e}). Falling back to sequential (may cause duplicates).")
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
                    event_tracks.append({
                        'particle_id': int(particle_id),
                        'pT': float(pt_map.get(particle_id, 0)),
                        'pdg_id': int(pdg_map.get(particle_id, 0)),
                        'points': points
                    })

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
            
            event_calo_hits = [
                {
                    'x': h['x'], 'y': h['y'], 'z': h['z'],
                    'energy': h['contrib_energies'],
                    'pT': h['pT'],
                    'pdg_id': int(h['pdg_id']),
                    'particle_id': int(h['contrib_particle_ids'])
                } for h in temp_calo_hits
            ]

        return {
            'tracks': event_tracks,
            'calo_hits': event_calo_hits,
            'all_tracker_hits': all_event_tracker_hits,
            'metadata': {
                "has_calo": bool(event_calo_hits),
                "has_pdg": True, # Assumed for now
                "has_volumes": has_volumes
            }
        }
