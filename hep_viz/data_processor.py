import os
from pathlib import Path
import pandas as pd
import numpy as np

import glob
import re

class DataProcessor:
    def __init__(self, path):
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
        Scan the directory and find the necessary files.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        # Define patterns to search for
        keys = ['particles', 'tracks', 'tracker_hits', 'calo_hits']
        
        if self.path.is_file():
            print(f"Warning: {self.path} is a file. Expecting a directory containing the dataset components.")
            return

        # Get all parquet files
        all_parquet = list(self.path.rglob("*.parquet"))
        all_parquet = [p for p in all_parquet if "val_loss" not in p.name]

        for key in keys:
            self.files[key] = []
            # 1. Try matching filename
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
                
                # Sort by start event
                self.files[key].sort(key=lambda x: x['start'])
            else:
                print(f"Warning: Could not find file for {key}")

    def _init_from_memory(self, data: dict):
        """
        Initialize from in-memory dictionary (e.g. Hugging Face dataset).
        """
        # data is expected to be {'particles': dataset, 'tracks': dataset, ...}
        self.memory_data = data
        
        # Create an index map for fast access if possible
        # Assuming HF dataset or list of dicts
        if 'particles' in self.memory_data:
            print("Indexing in-memory data...")
            # We need to know which row corresponds to which event_id
            # This assumes data is sorted or we scan it. 
            # For HF datasets, we can just iterate.
            try:
                # Check if it's a HF dataset (has features and rows)
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
        Load data for a specific event from a parquet file using filters.
        """
        try:
            # Use filters to load only the specific event
            df = pd.read_parquet(path, filters=[('event_id', '==', event_id)])
            
            # Explode common columns if they exist
            df = df.explode([col for col in df.columns if col != 'event_id'])
            
            if 'hit_ids' in df.columns:
                 df = df.explode('hit_ids')
            
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading event {event_id} from {path}: {e}")
            return pd.DataFrame()

    def get_event_list(self):
        """
        Return a list of available events.
        """
        if self.memory_data:
            events = sorted(list(self.event_index_map.keys()))
            return {"events": events}

        if 'particles' in self.files and self.files['particles']:
            all_events = set()
            try:
                for file_info in self.files['particles']:
                    # Optimization: if range is known, just add the range
                    if file_info['start'] != -1 and file_info['end'] != -1:
                        # end is usually inclusive in filenames like 0-9 (10 events)
                        # But let's be safe and read if unsure, or assume inclusive.
                        # Standard convention is inclusive.
                        all_events.update(range(file_info['start'], file_info['end'] + 1))
                    else:
                        # Fallback: read file
                        df = pd.read_parquet(file_info['path'], columns=['event_id'])
                        all_events.update(df['event_id'].unique().tolist())
                
                return {"events": sorted(list(all_events))}
            except Exception as e:
                print(f"Error reading event list: {e}")
                return {"events": []}
        return {"events": []}

    def process_event(self, event_id: str):
        """
        Load and process data for a specific event on demand.
        """
        try:
            n = int(event_id)
        except ValueError:
            raise ValueError(f"Invalid event ID: {event_id}")

        if self.memory_data:
            # Load from memory
            if n not in self.event_index_map:
                return {"error": f"Event {n} not found in memory data"}
            
            idx_map = self.event_index_map[n]
            
            # Particles
            if 'particles' not in idx_map:
                return {"error": "No particle data for this event"}
            
            p_idx = idx_map['particles']
            p_data = self.memory_data['particles'][p_idx]
            # Convert HF row (dict of lists/values) to DataFrame
            # HF row: {'event_id': 0, 'px': [1, 2], 'py': [3, 4], ...}
            # We need to convert this to a DataFrame where each particle is a row
            particles = pd.DataFrame(p_data)
            # If it's a single row (scalar values), wrap in list? 
            # HF datasets usually return a dict where values are lists if it's a list column.
            # But if it's a scalar column, it's a scalar.
            # Wait, the HF dataset structure for ColliderML has list columns for variable length data.
            # So p_data['px'] is a list of px values.
            # pd.DataFrame(p_data) should work if all lists have same length.
            
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
            # Load from files
            if 'particles' not in self.files or not self.files['particles']:
                return {"error": "No particle data found"}

            # Helper to find correct file
            def get_file_for_event(key, event_id):
                if key not in self.files:
                    return None
                
                # 1. Check ranges
                for f in self.files[key]:
                    if f['start'] <= event_id <= f['end']:
                        return f['path']
                
                # 2. If no range match (or ranges unknown), check all files (slow fallback)
                # Or maybe we just return None and fail?
                # Let's try to be robust: if ranges are -1, we might need to check.
                # But for now, let's assume if ranges are present, they are correct.
                # If ranges are -1, we pick the first one? No, that's bad.
                # If ranges are -1, we might have to open them.
                for f in self.files[key]:
                    if f['start'] == -1:
                         # Try loading? This is expensive.
                         # Maybe just return the first one if only one exists
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

        # Common processing (rest of the function)
        
        if not tracker_hits.empty:
             # Pre-process tracker hits (create HIT_ID) - doing this per event now
             if 'HIT_ID' not in tracker_hits.columns:
                tracker_hits = tracker_hits.reset_index(drop=True)
                tracker_hits = tracker_hits.reset_index().rename(columns={'index': 'HIT_ID'})
             tracker_hits = tracker_hits.apply(pd.to_numeric, errors='coerce')

        if not calo_hits.empty:
             # Pre-process calo hits
             if 'HIT_ID' not in calo_hits.columns:
                calo_hits = calo_hits.reset_index(drop=True)
                calo_hits = calo_hits.reset_index().rename(columns={'index': 'HIT_ID'})
             
             # Explode if needed (for file based it was needed, for HF it might be list of lists?)
             # If HF returns lists, pd.DataFrame(data) makes columns of lists.
             # We need to explode them?
             # Let's check. If I do pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), I get 2 rows.
             # If HF row is {'px': [1, 2], 'py': [3, 4]}, pd.DataFrame(row) gives 2 rows.
             # So for particles and tracker hits, it's fine.
             # For calo hits, we have nested lists? 
             # "Format: Apache Parquet with list columns for variable-length data"
             # So `contrib_particle_ids` is a list of lists? No, for a single event, it's a list of lists (one list per hit).
             # Wait, `_load_parquet_event` does `df.explode`.
             # If we load from HF, we get one "row" which is the event.
             # The columns are lists of values (e.g. x is list of floats).
             # So `pd.DataFrame(p_data)` creates a DF where each row is a hit/particle.
             # BUT, `contrib_particle_ids` for a calo hit is ITSELF a list.
             # So `pd.DataFrame(ch_data)` will have a column `contrib_particle_ids` where each element is a list.
             # So we DO need to explode it, just like in file loading.
             
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
                     calo_hits = calo_hits.explode(actual_explode_cols)
                 except ValueError as e:
                     # Fallback for older pandas or mismatched lengths (though shouldn't happen in valid data)
                     print(f"Warning: Simultaneous explode failed ({e}). Falling back to sequential (may cause duplicates).")
                     for col in actual_explode_cols:
                         calo_hits = calo_hits.explode(col)

             all_numeric_cols = ['cell_id', 'total_energy', 'x', 'y', 'z','contrib_particle_ids', 'contrib_energies', 'contrib_times']
             for col in all_numeric_cols:
                 if col in calo_hits.columns:
                     calo_hits[col] = pd.to_numeric(calo_hits[col], errors='coerce')

        # 2. Apply static cuts
        if 'vx' in particles.columns and 'vy' in particles.columns:
            particles = particles[abs(particles['vx']) < 1]
            particles = particles[abs(particles['vy']) < 1]
        
        # 3. Calculate pT and PDG ID
        if 'px' in particles.columns and 'py' in particles.columns:
            # Ensure numeric
            particles['px'] = pd.to_numeric(particles['px'], errors='coerce')
            particles['py'] = pd.to_numeric(particles['py'], errors='coerce')
            
            particles['pT'] = np.sqrt(particles['px']**2 + particles['py']**2)
        else:
            particles['pT'] = 0.0
        
        # Get particle IDs to match
        particles_id = particles["particle_id"].unique()
        
        # Create lookup maps
        pt_map = particles.set_index('particle_id')['pT'].to_dict()
        pdg_map = particles.set_index('particle_id')['pdg_id'].to_dict()

        # 4.1 Process ALL Tracker Hits
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
                # Ensure int
                filtered_tracker_hits['volume_id'] = pd.to_numeric(filtered_tracker_hits['volume_id'], errors='coerce').fillna(-1).astype(int)
            
            all_event_tracker_hits = filtered_tracker_hits[cols_to_keep].to_dict(orient='records')

        # 4.2 Process Tracks
        event_tracks = []
        if not tracker_hits.empty:
            tracker_hits_for_tracks = tracker_hits[tracker_hits["particle_id"].isin(particles_id)].copy()
            tracker_hits_for_tracks['r'] = np.sqrt(tracker_hits_for_tracks['x']**2 + tracker_hits_for_tracks['y']**2 + tracker_hits_for_tracks['z']**2)
            tracker_hits_for_tracks = tracker_hits_for_tracks.sort_values(by=['particle_id', 'r'])

            for particle_id, group_of_hits in tracker_hits_for_tracks.groupby('particle_id'):
                # Include volume_id in points if available
                cols = ['x', 'y', 'z']
                if 'volume_id' in group_of_hits.columns:
                    cols.append('volume_id')
                    # Ensure volume_id is int
                    group_of_hits['volume_id'] = pd.to_numeric(group_of_hits['volume_id'], errors='coerce').fillna(-1).astype(int)
                
                points = group_of_hits[cols].to_dict(orient='records')
                
                if len(points) > 1:
                    event_tracks.append({
                        'particle_id': int(particle_id),
                        'pT': float(pt_map.get(particle_id, 0)),
                        'pdg_id': int(pdg_map.get(particle_id, 0)),
                        'points': points
                    })

        # 5. Process Calo Hits
        event_calo_hits = []
        if not calo_hits.empty and 'contrib_particle_ids' in calo_hits.columns:
            # Note: calo_hits is already exploded in load_data (now in process_event)
            
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
