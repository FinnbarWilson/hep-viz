import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob

class DataProcessor:
    def __init__(self, path: str):
        self.path = Path(path)
        self.particles_df = None
        self.tracks_df = None
        self.tracker_hits_df = None
        self.calo_hits_df = None
        self.load_data()

    def load_data(self):
        """
        Scan the directory and load the necessary files.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        # Define patterns to search for
        # We look for files that either have the key in the filename OR are in a directory with the key
        keys = ['particles', 'tracks', 'tracker_hits', 'calo_hits']
        
        found_files = {}

        if self.path.is_file():
            print(f"Warning: {self.path} is a file. Expecting a directory containing the dataset components.")
            return

        # Get all parquet files
        all_parquet = list(self.path.rglob("*.parquet"))
        all_parquet = [p for p in all_parquet if "val_loss" not in p.name]

        for key in keys:
            # 1. Try matching filename
            matches = [p for p in all_parquet if key in p.name]
            
            # 2. If no filename match, try matching parent directory name
            if not matches:
                matches = [p for p in all_parquet if key in p.parent.name]
            
            if matches:
                # Take the first match
                found_files[key] = matches[0]
                print(f"Found {key}: {matches[0]}")
            else:
                print(f"Warning: Could not find file for {key}")

        # Load DataFrames
        if 'particles' in found_files:
            self.particles_df = self._load_parquet(found_files['particles'])
        
        if 'tracks' in found_files:
            self.tracks_df = self._load_parquet(found_files['tracks'])
            
        if 'tracker_hits' in found_files:
            self.tracker_hits_df = self._load_parquet(found_files['tracker_hits'])
            if self.tracker_hits_df is not None:
                 # Pre-process tracker hits (create HIT_ID)
                self.tracker_hits_df = self.tracker_hits_df.reset_index(drop=True)
                self.tracker_hits_df = self.tracker_hits_df.reset_index().rename(columns={'index': 'HIT_ID'})
                self.tracker_hits_df = self.tracker_hits_df.apply(pd.to_numeric, errors='coerce')
                self.tracker_hits_df = self.tracker_hits_df.set_index('HIT_ID')

        if 'calo_hits' in found_files:
            self.calo_hits_df = self._load_parquet(found_files['calo_hits'])
            if self.calo_hits_df is not None:
                # Pre-process calo hits
                self.calo_hits_df = self.calo_hits_df.reset_index(drop=True)
                self.calo_hits_df = self.calo_hits_df.reset_index().rename(columns={'index': 'HIT_ID'})
                self.calo_hits_df = self.calo_hits_df.explode(['contrib_particle_ids', 'contrib_energies', 'contrib_times'])
                self.calo_hits_df = self.calo_hits_df.set_index('HIT_ID')
                all_numeric_cols = ['cell_id', 'total_energy', 'x', 'y', 'z','contrib_particle_ids', 'contrib_energies', 'contrib_times']
                for col in all_numeric_cols:
                    if col in self.calo_hits_df.columns:
                        self.calo_hits_df[col] = pd.to_numeric(self.calo_hits_df[col], errors='coerce')


    def _load_parquet(self, path):
        try:
            df = pd.read_parquet(path)
            # Explode common columns if they exist
            explode_cols = [col for col in df.columns if col != 'event_id']
            # Check if columns are list-like before exploding? 
            # The reference code blindly explodes all except event_id.
            # But read_parquet might already handle some of this or return lists.
            # Let's follow reference logic but be careful.
            
            # Actually, read_parquet usually returns lists for repeated fields.
            # The reference code:
            # df = df.explode([col for col in df.columns if col != 'event_id'])
            # This implies that for a single row (event), all other columns are lists of the same length.
            
            df = df.explode([col for col in df.columns if col != 'event_id'])
            
            # Specific handling for tracks which has nested lists sometimes?
            # Reference: tracks = tracks.explode('hit_ids')
            if 'hit_ids' in df.columns:
                 df = df.explode('hit_ids')
            
            # Reference: apply(pd.to_numeric)
            # This might be slow for large DFs, but let's stick to it for now or optimize later.
            # We will do type conversion on demand or for specific columns to be safe.
            
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def get_event_list(self):
        """
        Return a list of available events.
        """
        if self.particles_df is not None:
            events = self.particles_df['event_id'].unique().tolist()
            return {"events": sorted(events)}
        return {"events": []}

    def process_event(self, event_id: str):
        """
        Load and process data for a specific event.
        """
        try:
            n = int(event_id)
        except ValueError:
            raise ValueError(f"Invalid event ID: {event_id}")

        if self.particles_df is None:
            return {"error": "No particle data found"}

        # Filter for event
        particles = self.particles_df[self.particles_df['event_id'] == n].copy()
        tracker_hits = self.tracker_hits_df[self.tracker_hits_df['event_id'] == n].copy() if self.tracker_hits_df is not None else pd.DataFrame()
        calo_hits = self.calo_hits_df[self.calo_hits_df['event_id'] == n].copy() if self.calo_hits_df is not None else pd.DataFrame()

        # 2. Apply static cuts
        if 'vx' in particles.columns and 'vy' in particles.columns:
            particles = particles[abs(particles['vx']) < 1]
            particles = particles[abs(particles['vy']) < 1]
        
        # 3. Calculate pT and PDG ID
        if 'px' in particles.columns and 'py' in particles.columns:
            # Ensure numeric
            particles['px'] = pd.to_numeric(particles['px'], errors='coerce')
            particles['py'] = pd.to_numeric(particles['py'], errors='coerce')
            
            # Debug
            # print(f"Particles dtypes:\n{particles.dtypes}")
            # print(f"PX head: {particles['px'].head()}")
            
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
            # Note: calo_hits is already exploded in load_data
            
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
